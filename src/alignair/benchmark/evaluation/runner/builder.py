from __future__ import annotations

from typing import Any

from ...core import criteria_catalog, metric_spec_catalog, scenario_axes_catalog, validate_catalogs
from ...core.artifacts import BENCHMARK_REPORT, artifact_metadata
from ...core.schema import BenchmarkCase
from ...generation import coverage_summary
from ..audit import audit_criteria_report
from ..contract import prediction_contract, validate_predictions, validate_predictions_for_cases
from ..diagnostics import build_allele_calling_diagnostics, build_boundary_diagnostics
from ..matching import align_predictions_to_cases
from ..scoring import audit_scoring_runtime, score_cases, scoring_manifest_catalog
from ..performance import (
    normalize_performance_summary,
    performance_metrics_from_summary,
    summarize_prediction_performance,
)
from ..report import build_assay_report
from ..uncertainty import bootstrap_metric_intervals

from .scoring import _overall_and_contexts


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
        "artifact": artifact_metadata(BENCHMARK_REPORT),
        "benchmark": {
            "n_cases": len(cases),
            "case_id_first": cases[0].case_id if cases else None,
            "case_id_last": cases[-1].case_id if cases else None,
            "strata": sorted({case.stratum for case in cases}),
        },
        "frame": frame,
        "criteria": criteria_catalog(),
        "catalog_validation": validate_catalogs(),
        "metric_registry": metric_spec_catalog(),
        "scoring_manifest": scoring_manifest_catalog(),
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
    report["scoring_audit"] = audit_scoring_runtime(report, cases=cases, predictions=predictions)
    report["criteria_audit"] = audit_criteria_report(report, cases=cases)
    report["assay"] = build_assay_report(report)
    return report
