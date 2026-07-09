from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core.schema import BenchmarkCase
from ..context import case_contexts
from ..scoring import score_cases


def _score_contexts(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    frame: str,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, case in enumerate(cases):
        for context in set(case_contexts(case)):
            grouped[context].append(idx)
    return {
        context: score_cases(
            [cases[i] for i in indices],
            [predictions[i] for i in indices],
            frame=frame,
            include_strata=False,
        )
        for context, indices in sorted(grouped.items())
    }


def _overall_and_contexts(
    scores: dict[str, Any],
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
    *,
    frame: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overall = dict(scores)
    overall.pop("by_stratum", {})
    by_context = _score_contexts(cases, predictions, frame=frame)
    return overall, by_context
