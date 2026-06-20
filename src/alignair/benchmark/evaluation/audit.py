"""Criteria and metric-coverage audits for benchmark reports."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from ..core import GENES, criteria_catalog
from ..core.schema import BenchmarkCase


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _extract_overall(scores_or_report: dict[str, Any] | None) -> dict[str, Any]:
    payload = scores_or_report or {}
    if "results" in payload:
        return payload.get("results", {}).get("overall", {})
    return payload


def _observed_metric_keys(scores_or_report: dict[str, Any] | None) -> dict[str, Any]:
    overall = _extract_overall(scores_or_report)
    global_keys = {
        key
        for key, value in overall.get("global", {}).items()
        if _finite(value) is not None
    }
    gene_keys: dict[str, set[str]] = {}
    for gene, metrics in overall.get("genes", {}).items():
        gene_keys[gene] = {
            key for key, value in metrics.items() if _finite(value) is not None
        }
    combined = set(global_keys)
    for keys in gene_keys.values():
        combined.update(keys)
    return {
        "global": sorted(global_keys),
        "genes": {gene: sorted(keys) for gene, keys in sorted(gene_keys.items())},
        "all": sorted(combined),
    }


def _case_truth_value(case: BenchmarkCase, field: str) -> Any:
    if field in {"case_id", "sequence_id"}:
        return case.case_id
    if field == "sequence":
        return case.sequence
    if field == "canonical_sequence":
        return case.canonical_sequence
    if field == "sequence_length":
        return len(case.sequence)
    if field == "orientation_id":
        return case.orientation_id
    if field == "rev_comp":
        return case.orientation_id in (1, 3)
    if field == "region_labels":
        return case.region_labels
    if field == "state_labels":
        return case.state_labels
    if field in case.record:
        return case.record.get(field)
    if field in case.scalars:
        return case.scalars.get(field)

    parts = field.split("_")
    if len(parts) >= 2 and parts[0] in GENES:
        gene = parts[0]
        suffix = "_".join(parts[1:])
        truth = case.genes.get(gene)
        if truth is None:
            return None
        if suffix == "call":
            return truth.calls
        if suffix == "sequence_start":
            return truth.sequence_start
        if suffix == "sequence_end":
            return truth.sequence_end
        if suffix == "germline_start":
            return truth.germline_start
        if suffix == "germline_end":
            return truth.germline_end
    return None


def _truth_field_availability(
    cases: Iterable[BenchmarkCase] | None,
    fields: Iterable[str],
) -> dict[str, dict[str, Any]]:
    case_list = list(cases or [])
    out = {}
    for field in sorted(set(fields)):
        present = 0
        for case in case_list:
            if not _is_missing(_case_truth_value(case, field)):
                present += 1
        out[field] = {
            "n_present": present,
            "n_cases": len(case_list),
            "fraction_present": (present / len(case_list)) if case_list else None,
        }
    return out


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
    criterion_metric_keys = {
        key for criterion in criteria for key in criterion.get("metric_keys", ())
    }
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
            "n_metric_keys_without_criteria": len(observed_all - criterion_metric_keys),
            "n_criteria_without_metric_coverage": len(criteria_without_metric_coverage),
            "n_available_but_unobserved": len(available_but_unobserved),
            "n_planned_but_observed": len(planned_but_observed),
            "n_available_truth_field_gaps": len(available_truth_field_gaps),
            "has_case_truth_audit": cases is not None,
        },
        "observed_metric_keys": observed,
        "metric_keys_without_criteria": sorted(observed_all - criterion_metric_keys),
        "catalog_metric_keys_without_values": sorted(criterion_metric_keys - observed_all),
        "available_but_unobserved": available_but_unobserved,
        "planned_but_observed": planned_but_observed,
        "criteria_without_metric_coverage": criteria_without_metric_coverage,
        "available_truth_field_gaps": available_truth_field_gaps,
        "truth_field_availability": truth_availability,
        "criteria": criteria_rows,
        "truth_source": "GenAIRR benchmark cases" if cases is not None else None,
    }
