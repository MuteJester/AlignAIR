from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...core import BENCHMARK_REPORT, metric_registry, validate_artifact
from .registry import (
    _observed_metric_keys,
    _criteria_metric_keys,
    _embedded_metric_registry,
    _embedded_scoring_manifest,
)
from .audits import (
    _audit_count_mismatches,
    _scoring_audit_count_mismatches,
)


def validate_benchmark_report_contract(
    report: Mapping[str, Any] | None,
    *,
    require_current_version: bool = False,
    require_metric_registry: bool = False,
) -> dict[str, Any]:
    """Validate benchmark report structure and embedded metric metadata.

    This is intentionally stricter than ``validate_artifact`` for reports while
    remaining extension-friendly: unknown observed metrics are warnings, not
    hard failures, unless the embedded registry itself is malformed.
    """

    payload = report or {}
    artifact = validate_artifact(
        payload,
        BENCHMARK_REPORT,
        require_current_version=require_current_version,
    )
    errors = list(artifact["problems"])
    warnings = []

    registry_keys, registry_problems, duplicate_registry_keys = _embedded_metric_registry(payload)
    for problem in registry_problems:
        if problem == "missing_metric_registry" and not require_metric_registry:
            warnings.append(problem)
        else:
            errors.append(problem)

    current_registry_keys = set(metric_registry())
    missing_current_registry_keys = current_registry_keys - registry_keys
    extra_registry_keys = registry_keys - current_registry_keys
    if registry_keys and (missing_current_registry_keys or extra_registry_keys):
        if require_current_version:
            errors.append("metric_registry_differs_from_current")
        else:
            warnings.append("metric_registry_differs_from_current")

    (
        scoring_manifest_metric_keys,
        scoring_manifest_problems,
        duplicate_scoring_components,
        scoring_manifest_without_registry,
    ) = _embedded_scoring_manifest(payload, registry_metric_keys=registry_keys)
    errors.extend(scoring_manifest_problems)

    criteria_metric_keys = _criteria_metric_keys(payload)
    observed_metric_keys = _observed_metric_keys(payload)
    unregistered_criteria_metric_keys = criteria_metric_keys - registry_keys
    observed_without_registry = observed_metric_keys - registry_keys
    if unregistered_criteria_metric_keys:
        errors.append("criteria_metric_keys_without_registry")
    if observed_without_registry:
        warnings.append("observed_metric_keys_without_registry")

    audit_mismatches = _audit_count_mismatches(
        payload,
        observed_metric_keys=observed_metric_keys,
        criteria_metric_keys=criteria_metric_keys,
        registry_metric_keys=registry_keys,
    )
    if audit_mismatches:
        errors.append("criteria_audit_count_mismatch")

    scoring_audit_mismatches = _scoring_audit_count_mismatches(
        payload,
        observed_metric_keys=observed_metric_keys,
        scoring_manifest_metric_keys=scoring_manifest_metric_keys,
    )
    if scoring_audit_mismatches:
        errors.append("scoring_audit_count_mismatch")

    errors = tuple(dict.fromkeys(errors))
    warnings = tuple(dict.fromkeys(warnings))
    return {
        "valid": not errors,
        "problems": errors,
        "warnings": warnings,
        "artifact": artifact,
        "metric_registry": {
            "present": "metric_registry" in payload,
            "n_report_metric_keys": len(registry_keys),
            "n_current_metric_keys": len(current_registry_keys),
            "duplicate_keys": duplicate_registry_keys,
            "missing_current_keys": tuple(sorted(missing_current_registry_keys)),
            "extra_keys": tuple(sorted(extra_registry_keys)),
        },
        "scoring_manifest": {
            "present": "scoring_manifest" in payload,
            "n_metric_keys": len(scoring_manifest_metric_keys),
            "duplicate_component_names": duplicate_scoring_components,
            "metric_keys_without_registry": scoring_manifest_without_registry,
        },
        "criteria": {
            "n_metric_keys": len(criteria_metric_keys),
            "unregistered_metric_keys": tuple(sorted(unregistered_criteria_metric_keys)),
        },
        "observed": {
            "n_metric_keys": len(observed_metric_keys),
            "metric_keys_without_registry": tuple(sorted(observed_without_registry)),
        },
        "criteria_audit_count_mismatches": audit_mismatches,
        "scoring_audit_count_mismatches": scoring_audit_mismatches,
    }
