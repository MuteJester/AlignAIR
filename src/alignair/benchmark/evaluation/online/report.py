from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from ...core import GENES, BenchmarkCase, BenchmarkSpec, criteria_catalog, metric_spec_catalog, scenario_axes_catalog, validate_catalogs
from ...core.artifacts import BENCHMARK_REPORT, artifact_metadata

from ..contract import prediction_contract
from ..context import case_contexts
from ..diagnostics import AlleleCallingDiagnosticsAccumulator, BoundaryDiagnosticsAccumulator
from ..scoring import score_one_case, scoring_manifest_catalog

from .accumulator import _Accumulator


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
            "artifact": artifact_metadata(BENCHMARK_REPORT),
            "benchmark": asdict(self.spec),
            "frame": self.frame,
            "criteria": criteria_catalog(),
            "catalog_validation": validate_catalogs(),
            "metric_registry": metric_spec_catalog(),
            "scoring_manifest": scoring_manifest_catalog(),
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
