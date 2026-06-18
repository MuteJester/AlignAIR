"""Generic benchmark runner utilities."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .metrics import score_cases
from ..core.schema import BenchmarkCase


Predictor = Callable[[list[str]], list[dict[str, Any]]]


def run_benchmark(
    cases: list[BenchmarkCase],
    predictor: Predictor,
    *,
    frame: str = "canonical",
) -> dict[str, Any]:
    """Run ``predictor`` on benchmark sequences and score the predictions."""

    predictions = predictor([c.sequence for c in cases])
    return score_cases(cases, predictions, frame=frame)
