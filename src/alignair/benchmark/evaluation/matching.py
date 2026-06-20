"""Prediction-to-case matching helpers for offline benchmark reports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.schema import BenchmarkCase


@dataclass(frozen=True)
class PredictionMatchResult:
    """Predictions aligned to benchmark case order plus match diagnostics."""

    predictions: list[dict[str, Any]]
    report: dict[str, Any]


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _case_id(case: BenchmarkCase, id_field: str) -> str:
    if id_field in {"case_id", "sequence_id"}:
        return case.case_id
    value = case.record.get(id_field)
    if not _missing(value):
        return str(value)
    value = getattr(case, id_field, None)
    if not _missing(value):
        return str(value)
    return case.case_id


def _prediction_id(prediction: dict[str, Any] | None, id_field: str) -> str | None:
    if not prediction:
        return None
    value = prediction.get(id_field)
    if _missing(value) and id_field == "sequence_id":
        value = prediction.get("case_id")
    if _missing(value) and id_field == "case_id":
        value = prediction.get("sequence_id")
    if _missing(value):
        return None
    return str(value)


def align_predictions_to_cases(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    id_field: str = "sequence_id",
    duplicate_policy: str = "first",
    max_examples: int = 25,
) -> PredictionMatchResult:
    """Align predictions to benchmark case order by a stable identifier.

    Missing case predictions are filled with ``{}`` so they are counted as
    failures by the scorer and by prediction-contract validation.
    """

    if duplicate_policy not in {"first", "last", "error"}:
        raise ValueError("duplicate_policy must be one of: first, last, error")

    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    without_id: list[int] = []
    for idx, prediction in enumerate(predictions):
        pid = _prediction_id(prediction, id_field)
        if pid is None:
            without_id.append(idx)
            continue
        buckets.setdefault(pid, []).append((idx, dict(prediction or {})))

    duplicate_ids = {pid: rows for pid, rows in buckets.items() if len(rows) > 1}
    if duplicate_ids and duplicate_policy == "error":
        ids = ", ".join(sorted(duplicate_ids)[:max_examples])
        raise ValueError(f"duplicate prediction ids for {id_field}: {ids}")

    selected: dict[str, dict[str, Any]] = {}
    for pid, rows in buckets.items():
        selected[pid] = rows[-1][1] if duplicate_policy == "last" else rows[0][1]

    case_ids = [_case_id(case, id_field) for case in cases]
    case_id_set = set(case_ids)
    aligned: list[dict[str, Any]] = []
    missing_case_ids: list[str] = []
    for cid in case_ids:
        pred = selected.get(cid)
        if pred is None:
            missing_case_ids.append(cid)
            aligned.append({})
        else:
            aligned.append(pred)

    extra_ids = sorted(pid for pid in selected if pid not in case_id_set)
    duplicate_items = [
        {"id": pid, "count": len(rows)}
        for pid, rows in sorted(duplicate_ids.items(), key=lambda item: item[0])
    ]
    matched = len(cases) - len(missing_case_ids)
    report = {
        "mode": "id",
        "id_field": id_field,
        "duplicate_policy": duplicate_policy,
        "n_cases": len(cases),
        "n_predictions": len(predictions),
        "n_matched_cases": matched,
        "match_fraction": matched / len(cases) if cases else 1.0,
        "n_missing_cases": len(missing_case_ids),
        "missing_case_ids": missing_case_ids[:max_examples],
        "n_extra_prediction_ids": len(extra_ids),
        "extra_prediction_ids": extra_ids[:max_examples],
        "n_duplicate_prediction_ids": len(duplicate_items),
        "duplicate_prediction_ids": duplicate_items[:max_examples],
        "n_predictions_without_id": len(without_id),
        "prediction_indices_without_id": without_id[:max_examples],
        "examples_truncated_to": max_examples,
    }
    return PredictionMatchResult(predictions=aligned, report=report)
