"""Benchmark generation recipes and GenAIRR builders."""

from .generate import coverage_summary, dataconfig_by_name, generate_benchmark, stream_benchmark
from .planner import (
    ALLELE_CONTEXT_PREFIX,
    CoverageGenerationResult,
    CoveragePlan,
    CoverageTracker,
    allele_context_label,
    allele_stratification_contexts,
    case_coverage_labels,
    core_context_min_counts,
    coverage_plan_from_spec,
    generate_coverage_benchmark,
    stream_coverage_benchmark,
)
from .readiness import ReadinessThresholds, assess_benchmark_readiness, readiness_thresholds
from .strata import default_igh_assay_spec, default_igh_spec, focused_igh_spec, focused_igh_strata

__all__ = [
    "CoverageGenerationResult",
    "CoveragePlan",
    "CoverageTracker",
    "ALLELE_CONTEXT_PREFIX",
    "ReadinessThresholds",
    "allele_context_label",
    "allele_stratification_contexts",
    "assess_benchmark_readiness",
    "case_coverage_labels",
    "core_context_min_counts",
    "coverage_summary",
    "coverage_plan_from_spec",
    "dataconfig_by_name",
    "default_igh_assay_spec",
    "default_igh_spec",
    "focused_igh_spec",
    "focused_igh_strata",
    "generate_benchmark",
    "generate_coverage_benchmark",
    "readiness_thresholds",
    "stream_benchmark",
    "stream_coverage_benchmark",
]
