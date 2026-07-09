from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _audit_count_mismatches(
    report: Mapping[str, Any],
    *,
    observed_metric_keys: set[str],
    criteria_metric_keys: set[str],
    registry_metric_keys: set[str],
) -> tuple[str, ...]:
    audit = report.get("criteria_audit")
    if not isinstance(audit, Mapping):
        return ()
    summary = audit.get("summary")
    if not isinstance(summary, Mapping):
        return ()

    expected = {
        "n_observed_metric_keys": len(observed_metric_keys),
        "n_catalog_metric_keys": len(criteria_metric_keys),
        "n_registered_metric_keys": len(registry_metric_keys),
    }
    mismatches = []
    for key, value in expected.items():
        if key in summary and summary.get(key) != value:
            mismatches.append(key)
    return tuple(mismatches)


def _scoring_audit_count_mismatches(
    report: Mapping[str, Any],
    *,
    observed_metric_keys: set[str],
    scoring_manifest_metric_keys: set[str],
) -> tuple[str, ...]:
    audit = report.get("scoring_audit")
    if not isinstance(audit, Mapping):
        return ()
    summary = audit.get("summary")
    if not isinstance(summary, Mapping):
        return ()

    expected = {
        "n_observed_metric_keys": len(observed_metric_keys),
        "n_declared_metric_keys": len(scoring_manifest_metric_keys),
        "n_observed_declared_metric_keys": len(observed_metric_keys & scoring_manifest_metric_keys),
        "n_observed_metric_keys_without_manifest": len(observed_metric_keys - scoring_manifest_metric_keys),
    }
    mismatches = []
    for key, value in expected.items():
        if key in summary and summary.get(key) != value:
            mismatches.append(key)
    return tuple(mismatches)
