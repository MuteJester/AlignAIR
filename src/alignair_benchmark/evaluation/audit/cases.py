from __future__ import annotations

from typing import Any, Iterable

from ...core import GENES
from ...core.schema import BenchmarkCase


def _is_missing(value: Any) -> bool:
    import math
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


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
    if field == "presented_region_labels":
        return case.presented_region_labels
    if field == "state_labels":
        return case.state_labels
    if field == "presented_state_labels":
        return case.presented_state_labels
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


def case_truth_value(case: BenchmarkCase, field: str) -> Any:
    """Return a benchmark truth value by stable field name."""

    return _case_truth_value(case, field)


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


def truth_field_availability(
    cases: Iterable[BenchmarkCase] | None,
    fields: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Return truth-field availability counts for benchmark cases."""

    return _truth_field_availability(cases, fields)
