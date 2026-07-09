from __future__ import annotations

import math
from typing import Any

from ...core.metric_registry import grade_metric_value, metric_higher_is_better, metric_quality

_GRADE_RANK = {"pass": 0, "warn": 1, "fail": 2, "not_scored": 3, "planned": 4}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _higher_is_better(metric: str) -> bool:
    return metric_higher_is_better(metric)


def _quality(metric: str, value: float) -> float:
    return metric_quality(metric, value)


def _grade_metric(metric: str, value: float) -> dict[str, Any]:
    return grade_metric_value(metric, value)


def _worst_grade(grades: list[str]) -> str:
    if not grades:
        return "not_scored"
    return max(grades, key=lambda grade: _GRADE_RANK.get(grade, -1))


def _metric_assessments(observed: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, values in observed.items():
        global_value = _finite(values.get("global"))
        if global_value is not None:
            rows.append(
                {
                    "metric": metric,
                    "scope": "global",
                    "gene": None,
                    "value": global_value,
                    **_grade_metric(metric, global_value),
                }
            )
        for gene, value in values.get("genes", {}).items():
            f = _finite(value)
            if f is None:
                continue
            rows.append(
                {
                    "metric": metric,
                    "scope": "gene",
                    "gene": gene,
                    "value": f,
                    **_grade_metric(metric, f),
                }
            )
    return rows


def _criterion_grade(
    *,
    status: str | None,
    coverage_fraction: float,
    missing_metric_keys: tuple[str, ...],
    metric_assessments: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    from collections import Counter
    if not metric_assessments:
        if status == "planned":
            return "planned", ["criterion is planned and has no implemented metrics yet"]
        return "not_scored", ["no metric values were observed for this criterion"]

    grades = [row["grade"] for row in metric_assessments]
    grade = _worst_grade(grades)
    reasons: list[str] = []
    counts = Counter(grades)
    if counts.get("fail"):
        reasons.append(f"{counts['fail']} metric value(s) below fail threshold")
    if counts.get("warn"):
        reasons.append(f"{counts['warn']} metric value(s) in warning range")
    if missing_metric_keys and status == "available":
        if coverage_fraction < 0.5:
            grade = "fail"
            reasons.append("less than half of expected metric keys are present")
        elif grade == "pass":
            grade = "warn"
            reasons.append("some expected metric keys are missing")
    if not reasons:
        reasons.append("all observed metric values passed configured thresholds")
    return grade, reasons
