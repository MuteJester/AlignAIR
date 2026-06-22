"""Generic benchmark runner utilities."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .audit import audit_criteria_report
from .contract import prediction_contract, validate_predictions, validate_predictions_for_cases
from .context import case_contexts
from .diagnostics import build_allele_calling_diagnostics, build_boundary_diagnostics
from .matching import align_predictions_to_cases
from .metrics import score_cases
from .performance import (
    normalize_performance_summary,
    performance_metrics_from_summary,
    profile_predictor_call,
    summarize_prediction_performance,
)
from .report import build_assay_report
from .uncertainty import bootstrap_metric_intervals
from ..core import criteria_catalog, scenario_axes_catalog
from ..core.schema import BenchmarkCase
from ..generation import coverage_summary


Predictor = Callable[[list[str]], list[dict[str, Any]]]


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


def build_benchmark_report(
    cases: list[BenchmarkCase],
    predictions: list[dict[str, Any] | None],
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
    performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score saved predictions and return the full benchmark report shape."""

    match_report = None
    if match_by and match_by != "order":
        match_result = align_predictions_to_cases(
            cases,
            predictions,
            id_field=match_by,
            duplicate_policy=duplicate_policy,
        )
        predictions = match_result.predictions
        match_report = match_result.report
    elif len(cases) != len(predictions):
        raise ValueError(f"case/prediction length mismatch: {len(cases)} != {len(predictions)}")
    case_aware_contract = has_d is None
    resolved_has_d = bool(has_d) if has_d is not None else any(case.genes.get("d") and case.genes["d"].calls for case in cases)
    scores = score_cases(cases, predictions, frame=frame)
    overall, by_context = _overall_and_contexts(scores, cases, predictions, frame=frame)
    if performance is not None:
        performance_summary = normalize_performance_summary(
            performance,
            n_sequences=len(cases),
            source=performance.get("source"),
        )
    else:
        performance_summary = summarize_prediction_performance(predictions, n_sequences=len(cases))
    if performance_summary is not None:
        overall.setdefault("global", {}).update(performance_metrics_from_summary(performance_summary))
    report = {
        "benchmark": {
            "n_cases": len(cases),
            "case_id_first": cases[0].case_id if cases else None,
            "case_id_last": cases[-1].case_id if cases else None,
            "strata": sorted({case.stratum for case in cases}),
        },
        "frame": frame,
        "criteria": criteria_catalog(),
        "prediction_contract": prediction_contract(),
        "scenario_axes": scenario_axes_catalog(),
        "coverage": coverage_summary(cases),
        "results": {
            "overall": overall,
            "by_context": by_context,
        },
        "diagnostics": {
            "allele_calling": build_allele_calling_diagnostics(
                cases,
                predictions,
                frame=frame,
                top_n=diagnostic_top_n,
                examples_per_row=diagnostic_examples,
            ),
            "boundaries": build_boundary_diagnostics(
                cases,
                predictions,
                frame=frame,
                top_n=diagnostic_top_n,
                examples_per_row=diagnostic_examples,
            ),
        },
    }
    if performance_summary is not None:
        report["performance"] = performance_summary
    if match_report is not None:
        report["prediction_matching"] = match_report
    if contract_level is not None:
        if case_aware_contract:
            report["prediction_validation"] = validate_predictions_for_cases(
                cases,
                predictions,
                level=contract_level,
            )
        else:
            report["prediction_validation"] = validate_predictions(
                predictions,
                level=contract_level,
                has_d=resolved_has_d,
            )
    if n_bootstrap:
        report["uncertainty"] = bootstrap_metric_intervals(
            cases,
            predictions,
            frame=frame,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=bootstrap_seed,
            include_strata=bootstrap_strata,
        )
    report["criteria_audit"] = audit_criteria_report(report, cases=cases)
    report["assay"] = build_assay_report(report)
    return report


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
