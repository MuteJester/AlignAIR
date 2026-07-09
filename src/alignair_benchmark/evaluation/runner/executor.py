from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...core.schema import BenchmarkCase
from ..scoring import score_cases
from ..performance import profile_predictor_call

from .builder import build_benchmark_report

Predictor = Callable[[list[str]], list[dict[str, Any]]]


def run_benchmark(
    cases: list[BenchmarkCase],
    predictor: Predictor,
    *,
    frame: str = "canonical",
    profile_runtime: bool = False,
    profile_memory: bool = True,
) -> dict[str, Any]:
    """Run ``predictor`` on benchmark sequences and score the predictions."""

    if profile_runtime:
        predictions, _ = profile_predictor_call(
            predictor,
            [c.sequence for c in cases],
            profile_memory=profile_memory,
        )
    else:
        predictions = predictor([c.sequence for c in cases])
    return score_cases(cases, predictions, frame=frame)


def run_benchmark_report(
    cases: list[BenchmarkCase],
    predictor: Predictor,
    *,
    frame: str = "canonical",
    contract_level: str | None = None,
    has_d: bool | None = None,
    match_by: str | None = None,
    duplicate_policy: str = "first",
    n_bootstrap: int = 0,
    confidence: float = 0.95,
    bootstrap_seed: int = 123,
    bootstrap_strata: bool = True,
    diagnostic_top_n: int = 20,
    diagnostic_examples: int = 5,
    profile_runtime: bool = True,
    profile_memory: bool = True,
) -> dict[str, Any]:
    """Run ``predictor`` and return the full benchmark report shape."""

    performance = None
    if profile_runtime:
        predictions, performance = profile_predictor_call(
            predictor,
            [c.sequence for c in cases],
            profile_memory=profile_memory,
        )
    else:
        predictions = predictor([c.sequence for c in cases])
    return build_benchmark_report(
        cases,
        predictions,
        frame=frame,
        contract_level=contract_level,
        has_d=has_d,
        match_by=match_by,
        duplicate_policy=duplicate_policy,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
        bootstrap_strata=bootstrap_strata,
        diagnostic_top_n=diagnostic_top_n,
        diagnostic_examples=diagnostic_examples,
        performance=performance,
    )
