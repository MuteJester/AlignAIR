from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase
from ..scoring import score_cases
from .math_utils import _finite


def _metric_value(scores: dict[str, Any], path: str) -> float | None:
    parts = path.split(".")
    if len(parts) == 2 and parts[0] == "global":
        return _finite(scores.get("global", {}).get(parts[1]))
    if len(parts) == 3 and parts[0] == "genes":
        return _finite(scores.get("genes", {}).get(parts[1], {}).get(parts[2]))
    return None


def _score_metric_values(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    frame: str,
    metric_paths: tuple[str, ...],
) -> dict[str, float]:
    scores = score_cases(
        cases,
        predictions,
        frame=frame,
        include_strata=False,
        include_expensive_record_fields=False,
    )
    values = {}
    for path in metric_paths:
        value = _metric_value(scores, path)
        if value is not None:
            values[path] = value
    return values
