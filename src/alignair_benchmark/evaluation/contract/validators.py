from __future__ import annotations

import math
from typing import Any

from ...core.schema import BenchmarkCase
from .fields import PREDICTION_FIELDS, _expected_fields


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool) or _is_missing(value):
        return False
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _is_float_like(value: Any) -> bool:
    if isinstance(value, bool) or _is_missing(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and value in (0, 1):
        return True
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "t", "f", "yes", "no", "1", "0"}:
        return True
    return False


def _matches_dtype(value: Any, dtype: str) -> bool:
    if _is_missing(value):
        return True
    if dtype == "str":
        return isinstance(value, str)
    if dtype == "list[str]":
        return isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
    if dtype == "list[int]":
        return isinstance(value, (list, tuple)) and all(_is_int_like(v) for v in value)
    if dtype == "int":
        return _is_int_like(value)
    if dtype == "float":
        return _is_float_like(value)
    if dtype == "bool":
        return _is_bool_like(value)
    if dtype == "bool|float":
        return _is_bool_like(value) or _is_float_like(value)
    if dtype == "int[0..3]":
        return _is_int_like(value) and 0 <= int(float(value)) <= 3
    return True


def validate_prediction(
    prediction: dict[str, Any] | None,
    *,
    level: str = "core",
    has_d: bool = True,
) -> dict[str, Any]:
    """Validate one normalized prediction dictionary against a contract level."""

    pred = prediction or {}
    fields = _expected_fields(level, has_d)
    expected_names = [field.name for field in fields]
    missing = [field.name for field in fields if _is_missing(pred.get(field.name))]
    malformed = []
    for field in fields:
        if field.name in pred and not _matches_dtype(pred[field.name], field.dtype):
            malformed.append({"field": field.name, "expected": field.dtype, "value_type": type(pred[field.name]).__name__})
    present = [name for name in expected_names if name not in missing]
    return {
        "valid": not missing and not malformed,
        "level": level,
        "has_d": has_d,
        "n_expected": len(expected_names),
        "n_present": len(present),
        "coverage_fraction": len(present) / len(expected_names) if expected_names else 1.0,
        "present_fields": present,
        "missing_fields": missing,
        "malformed_fields": malformed,
        "extra_fields": sorted(k for k in pred if k not in {field.name for field in PREDICTION_FIELDS}),
    }


def _case_has_d(case: BenchmarkCase) -> bool:
    return bool(case.genes.get("d") and case.genes["d"].calls)


def _summarize_validation_rows(
    rows: list[dict[str, Any]],
    *,
    level: str,
    has_d: bool | str,
    case_aware: bool = False,
    has_d_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    missing_counts: dict[str, int] = {}
    malformed_counts: dict[str, int] = {}
    for row in rows:
        for field in row["missing_fields"]:
            missing_counts[field] = missing_counts.get(field, 0) + 1
        for item in row["malformed_fields"]:
            field = item["field"]
            malformed_counts[field] = malformed_counts.get(field, 0) + 1
    return {
        "level": level,
        "has_d": has_d,
        "case_aware": case_aware,
        "has_d_counts": has_d_counts,
        "n_predictions": len(rows),
        "n_valid": sum(1 for row in rows if row["valid"]),
        "valid_fraction": (sum(1 for row in rows if row["valid"]) / len(rows)) if rows else 1.0,
        "mean_coverage_fraction": (
            sum(row["coverage_fraction"] for row in rows) / len(rows)
            if rows
            else 1.0
        ),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "malformed_field_counts": dict(sorted(malformed_counts.items())),
        "rows": rows,
    }


def validate_predictions(
    predictions: list[dict[str, Any] | None],
    *,
    level: str = "core",
    has_d: bool = True,
) -> dict[str, Any]:
    """Validate a batch of normalized prediction dictionaries."""

    rows = [validate_prediction(pred, level=level, has_d=has_d) for pred in predictions]
    return _summarize_validation_rows(rows, level=level, has_d=has_d)


def validate_predictions_for_cases(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    level: str = "core",
) -> dict[str, Any]:
    """Validate predictions using each benchmark case's D-segment requirement."""

    if len(cases) != len(predictions):
        raise ValueError(f"case/prediction length mismatch: {len(cases)} != {len(predictions)}")
    rows = [
        validate_prediction(pred, level=level, has_d=_case_has_d(case))
        for case, pred in zip(cases, predictions)
    ]
    has_d_count = sum(1 for case in cases if _case_has_d(case))
    return _summarize_validation_rows(
        rows,
        level=level,
        has_d="per_case",
        case_aware=True,
        has_d_counts={"has_d": has_d_count, "no_d": len(cases) - has_d_count},
    )
