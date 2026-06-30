from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ...core import BenchmarkCase, BenchmarkSpec
from ...generation import (
    CoveragePlan,
    CoverageTracker,
    dataconfig_by_name,
    stream_benchmark,
    stream_coverage_benchmark,
)
from ..contract import PredictionValidationAccumulator
from ..audit import audit_criteria_report
from ..scoring import audit_scoring_runtime
from ..performance import PerformanceAccumulator, performance_metrics_from_summary, profile_predictor_call
from ..report import build_assay_report

from .report import OnlineBenchmarkReport

Predictor = Callable[[list[str]], list[dict[str, Any]]]


def _batched(items: Iterable[BenchmarkCase], batch_size: int) -> Iterable[list[BenchmarkCase]]:
    batch: list[BenchmarkCase] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_online_benchmark(
    spec: BenchmarkSpec,
    predictor: Predictor,
    *,
    dataconfig=None,
    reference_set=None,
    batch_size: int = 64,
    frame: str = "canonical",
    coverage_plan: CoveragePlan | None = None,
    contract_level: str | None = None,
    profile_runtime: bool = True,
    profile_memory: bool = True,
) -> dict[str, Any]:
    """Generate cases online, call ``predictor`` in batches, and return an assay report."""

    if dataconfig is None:
        dataconfig = dataconfig_by_name(spec.dataconfig_name)
    if reference_set is None:
        from ....reference.reference_set import ReferenceSet

        reference_set = ReferenceSet.from_dataconfigs(dataconfig)
    report = OnlineBenchmarkReport(spec, frame=frame, reference_set=reference_set)
    validation = (
        PredictionValidationAccumulator(level=contract_level, has_d=None)
        if contract_level
        else None
    )
    performance_accumulator = PerformanceAccumulator() if profile_runtime else None
    coverage_tracker = None
    if coverage_plan is not None:
        coverage_tracker = CoverageTracker(coverage_plan)
        case_iter = stream_coverage_benchmark(
            spec,
            dataconfig=dataconfig,
            reference_set=reference_set,
            plan=coverage_plan,
            tracker=coverage_tracker,
        )
    else:
        case_iter = stream_benchmark(spec, dataconfig=dataconfig, reference_set=reference_set)
    for cases in _batched(case_iter, batch_size):
        if profile_runtime:
            preds, performance = profile_predictor_call(
                predictor,
                [c.sequence for c in cases],
                profile_memory=profile_memory,
            )
            if performance_accumulator is not None:
                performance_accumulator.update(performance)
        else:
            preds = predictor([c.sequence for c in cases])
        if len(preds) != len(cases):
            raise ValueError(f"predictor returned {len(preds)} predictions for {len(cases)} cases")
        if validation is not None:
            validation.update_for_cases(cases, preds)
        for case, pred in zip(cases, preds):
            report.update(case, pred)
    out = report.to_dict()
    performance_summary = performance_accumulator.to_dict() if performance_accumulator is not None else None
    if performance_summary is not None:
        out["performance"] = performance_summary
        out["results"]["overall"].setdefault("global", {}).update(
            performance_metrics_from_summary(performance_summary)
        )
    if coverage_tracker is not None:
        out["generation_coverage"] = coverage_tracker.to_dict()
    if validation is not None:
        out["prediction_validation"] = validation.to_dict()
    out["scoring_audit"] = audit_scoring_runtime(out)
    out["criteria_audit"] = audit_criteria_report(out)
    out["assay"] = build_assay_report(out)
    return out
