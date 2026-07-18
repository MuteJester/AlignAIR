"""Read-orientation scoring."""
from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase


def score_orientation(pred: dict[str, Any], case: BenchmarkCase) -> dict[str, float]:
    if "orientation_id" not in pred or pred["orientation_id"] is None:
        return {}
    return {
        "orientation_acc": 1.0 if int(pred["orientation_id"]) == int(case.orientation_id) else 0.0,
    }
