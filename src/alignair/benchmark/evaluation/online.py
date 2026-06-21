"""Online benchmark evaluation and assay-style reports."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any

from ..core import GENES, BenchmarkCase, BenchmarkSpec, criteria_catalog, scenario_axes_catalog
from ..generation import (
    CoveragePlan,
    CoverageTracker,
    dataconfig_by_name,
    stream_benchmark,
    stream_coverage_benchmark,
)
from .contract import PredictionValidationAccumulator, prediction_contract
from .context import case_contexts
from .audit import audit_criteria_report
from .diagnostics import AlleleCallingDiagnosticsAccumulator, BoundaryDiagnosticsAccumulator
from .metrics import score_one_case
from .report import build_assay_report

Predictor = Callable[[list[str]], list[dict[str, Any]]]


class _Accumulator:
    def __init__(self) -> None:
        self.n_cases = 0
        self.global_sums: dict[str, float] = defaultdict(float)
        self.global_counts: dict[str, int] = defaultdict(int)
        self.gene_sums: dict[str, dict[str, float]] = {g: defaultdict(float) for g in GENES}
        self.gene_counts: dict[str, dict[str, int]] = {g: defaultdict(int) for g in GENES}

    def update(self, scored: dict[str, Any]) -> None:
        self.n_cases += 1
        for k, v in scored.get("global", {}).items():
            self.global_sums[k] += float(v)
            self.global_counts[k] += 1
        for gene, vals in scored.get("genes", {}).items():
            if gene not in self.gene_sums:
                self.gene_sums[gene] = defaultdict(float)
                self.gene_counts[gene] = defaultdict(int)
            for k, v in vals.items():
                self.gene_sums[gene][k] += float(v)
                self.gene_counts[gene][k] += 1

    @staticmethod
    def _finalize(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float | None]:
        return {k: (sums[k] / counts[k] if counts[k] else None) for k in sorted(sums)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "global": self._finalize(self.global_sums, self.global_counts),
            "genes": {
                g: self._finalize(self.gene_sums.get(g, {}), self.gene_counts.get(g, {}))
                for g in GENES
            },
        }


class OnlineBenchmarkReport:
    """Streaming accumulator for benchmark results and coverage."""

    def __init__(self, spec: BenchmarkSpec, frame: str = "canonical", reference_set=None) -> None:
        self.spec = spec
        self.frame = frame
        self.reference_set = reference_set
        self.overall = _Accumulator()
        self.groups: dict[str, _Accumulator] = defaultdict(_Accumulator)
        self.allele_diagnostics = AlleleCallingDiagnosticsAccumulator(frame=frame)
        self.boundary_diagnostics = BoundaryDiagnosticsAccumulator(frame=frame)
        self.coverage = {
            "n_cases": 0,
            "by_stratum": Counter(),
            "contexts": Counter(),
            "orientation": Counter(),
            "length": {"min": None, "max": None, "sum": 0.0},
            "alleles": {g: Counter() for g in GENES},
            "multi_call_cases": Counter(),
        }

    def _update_coverage(self, case: BenchmarkCase, contexts: list[str]) -> None:
        self.coverage["n_cases"] += 1
        self.coverage["by_stratum"][case.stratum] += 1
        self.coverage["orientation"][case.orientation_id] += 1
        length = len(case.sequence)
        length_cov = self.coverage["length"]
        length_cov["min"] = length if length_cov["min"] is None else min(length_cov["min"], length)
        length_cov["max"] = length if length_cov["max"] is None else max(length_cov["max"], length)
        length_cov["sum"] += length
        for ctx in contexts:
            self.coverage["contexts"][ctx] += 1
        for g, truth in case.genes.items():
            if len(truth.calls) > 1:
                self.coverage["multi_call_cases"][g] += 1
            for call in truth.calls:
                self.coverage["alleles"][g][call] += 1

    def update(self, case: BenchmarkCase, pred: dict[str, Any] | None) -> None:
        scored = score_one_case(case, pred, frame=self.frame)
        contexts = case_contexts(case)
        self.overall.update(scored)
        self.allele_diagnostics.update(case, pred)
        self.boundary_diagnostics.update(case, pred)
        for ctx in contexts:
            self.groups[ctx].update(scored)
        self._update_coverage(case, contexts)

    def coverage_dict(self) -> dict[str, Any]:
        length = self.coverage["length"]
        n_cases = max(int(self.coverage["n_cases"]), 1)
        allele_summary = {}
        for g, counts in self.coverage["alleles"].items():
            total = None
            missing = []
            if self.reference_set is not None and g.upper() in self.reference_set.genes:
                ref = self.reference_set.gene(g.upper())
                total = len(ref.names)
                missing = [name for name in ref.names if counts.get(name, 0) == 0]
            allele_summary[g] = {
                "n_total_reference": total,
                "n_observed": len(counts),
                "fraction_observed": (len(counts) / total) if total else None,
                "n_missing": len(missing),
                "missing": missing[:50],
                "min_count": min(counts.values()) if counts else 0,
                "max_count": max(counts.values()) if counts else 0,
                "mean_count": (sum(counts.values()) / len(counts)) if counts else 0.0,
            }
        return {
            "n_cases": self.coverage["n_cases"],
            "by_stratum": dict(self.coverage["by_stratum"]),
            "contexts": dict(self.coverage["contexts"]),
            "orientation": dict(self.coverage["orientation"]),
            "length": {
                "min": length["min"],
                "max": length["max"],
                "mean": length["sum"] / n_cases,
            },
            "alleles": allele_summary,
            "multi_call_cases": dict(self.coverage["multi_call_cases"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": asdict(self.spec),
            "frame": self.frame,
            "criteria": criteria_catalog(),
            "prediction_contract": prediction_contract(),
            "scenario_axes": scenario_axes_catalog(),
            "coverage": self.coverage_dict(),
            "results": {
                "overall": self.overall.to_dict(),
                "by_context": {
                    name: acc.to_dict()
                    for name, acc in sorted(self.groups.items())
                    if acc.n_cases > 0
                },
            },
            "diagnostics": {
                "allele_calling": self.allele_diagnostics.to_dict(),
                "boundaries": self.boundary_diagnostics.to_dict(),
            },
        }


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
) -> dict[str, Any]:
    """Generate cases online, call ``predictor`` in batches, and return an assay report."""

    if dataconfig is None:
        dataconfig = dataconfig_by_name(spec.dataconfig_name)
    if reference_set is None:
        from ...reference.reference_set import ReferenceSet

        reference_set = ReferenceSet.from_dataconfigs(dataconfig)
    report = OnlineBenchmarkReport(spec, frame=frame, reference_set=reference_set)
    validation = (
        PredictionValidationAccumulator(level=contract_level, has_d=None)
        if contract_level
        else None
    )
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
        preds = predictor([c.sequence for c in cases])
        if len(preds) != len(cases):
            raise ValueError(f"predictor returned {len(preds)} predictions for {len(cases)} cases")
        if validation is not None:
            validation.update_for_cases(cases, preds)
        for case, pred in zip(cases, preds):
            report.update(case, pred)
    out = report.to_dict()
    if coverage_tracker is not None:
        out["generation_coverage"] = coverage_tracker.to_dict()
    if validation is not None:
        out["prediction_validation"] = validation.to_dict()
    out["criteria_audit"] = audit_criteria_report(out)
    out["assay"] = build_assay_report(out)
    return out
