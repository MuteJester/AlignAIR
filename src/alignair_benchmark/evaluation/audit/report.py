from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from ...core import criteria_catalog, metric_registry
from ...core.schema import BenchmarkCase

from .cases import _truth_field_availability
from .metrics import _observed_metric_keys, _audit_metric_spec


def audit_criteria_report(
    scores_or_report: dict[str, Any] | None = None,
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    criteria: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit criteria status against observed metrics and GenAIRR truth fields."""

    criteria = criteria or (scores_or_report or {}).get("criteria") or criteria_catalog()
    observed = _observed_metric_keys(scores_or_report)
    observed_all = set(observed["all"])
    registry = metric_registry()
    registered_metric_keys = set(registry)
    criterion_metric_keys = {
        key for criterion in criteria for key in criterion.get("metric_keys", ())
    }
    unregistered_catalog_metric_keys = criterion_metric_keys - registered_metric_keys
    observed_metric_keys_without_registry = observed_all - registered_metric_keys
    all_truth_fields = {
        field for criterion in criteria for field in criterion.get("ground_truth_fields", ())
    }
    truth_availability = _truth_field_availability(cases, all_truth_fields)

    criteria_rows = []
    available_but_unobserved = []
    planned_but_observed = []
    criteria_without_metric_coverage = []
    available_truth_field_gaps = []
    status_counts = Counter()
    for criterion in criteria:
        status = criterion.get("status", "available")
        status_counts[status] += 1
        metric_keys = tuple(criterion.get("metric_keys", ()))
        row_registered_metric_keys = tuple(
            key for key in metric_keys if key in registered_metric_keys
        )
        row_unregistered_metric_keys = tuple(
            key for key in metric_keys if key not in registered_metric_keys
        )
        observed_keys = tuple(key for key in metric_keys if key in observed_all)
        missing_keys = tuple(key for key in metric_keys if key not in observed_all)
        truth_fields = tuple(criterion.get("ground_truth_fields", ()))
        unavailable_truth = tuple(
            field
            for field in truth_fields
            if truth_availability.get(field, {}).get("n_present", 0) == 0
        )
        row = {
            "name": criterion["name"],
            "category": criterion["category"],
            "status": status,
            "importance": criterion.get("importance"),
            "metric_keys": metric_keys,
            "registered_metric_keys": row_registered_metric_keys,
            "unregistered_metric_keys": row_unregistered_metric_keys,
            "metric_specs": tuple(_audit_metric_spec(key) for key in metric_keys),
            "observed_metric_keys": observed_keys,
            "missing_metric_keys": missing_keys,
            "metric_coverage_fraction": len(observed_keys) / len(metric_keys) if metric_keys else 0.0,
            "ground_truth_fields": truth_fields,
            "unavailable_truth_fields": unavailable_truth,
        }
        criteria_rows.append(row)
        if not observed_keys:
            criteria_without_metric_coverage.append(criterion["name"])
            if status == "available":
                available_but_unobserved.append(criterion["name"])
        if status == "planned" and observed_keys:
            planned_but_observed.append(criterion["name"])
        if status == "available" and unavailable_truth:
            available_truth_field_gaps.append(
                {
                    "criterion": criterion["name"],
                    "missing_truth_fields": unavailable_truth,
                }
            )

    return {
        "summary": {
            "n_criteria": len(criteria),
            "status_counts": dict(sorted(status_counts.items())),
            "n_observed_metric_keys": len(observed_all),
            "n_catalog_metric_keys": len(criterion_metric_keys),
            "n_registered_metric_keys": len(registered_metric_keys),
            "n_unregistered_catalog_metric_keys": len(unregistered_catalog_metric_keys),
            "n_observed_metric_keys_without_registry": len(observed_metric_keys_without_registry),
            "n_metric_keys_without_criteria": len(observed_all - criterion_metric_keys),
            "n_criteria_without_metric_coverage": len(criteria_without_metric_coverage),
            "n_available_but_unobserved": len(available_but_unobserved),
            "n_planned_but_observed": len(planned_but_observed),
            "n_available_truth_field_gaps": len(available_truth_field_gaps),
            "has_case_truth_audit": cases is not None,
        },
        "observed_metric_keys": observed,
        "metric_keys_without_criteria": sorted(observed_all - criterion_metric_keys),
        "observed_metric_keys_without_registry": sorted(observed_metric_keys_without_registry),
        "catalog_metric_keys_without_values": sorted(criterion_metric_keys - observed_all),
        "unregistered_catalog_metric_keys": sorted(unregistered_catalog_metric_keys),
        "available_but_unobserved": available_but_unobserved,
        "planned_but_observed": planned_but_observed,
        "criteria_without_metric_coverage": criteria_without_metric_coverage,
        "available_truth_field_gaps": available_truth_field_gaps,
        "truth_field_availability": truth_availability,
        "criteria": criteria_rows,
        "truth_source": "GenAIRR benchmark cases" if cases is not None else None,
    }
