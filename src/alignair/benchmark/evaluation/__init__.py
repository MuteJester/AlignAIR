"""Prediction adapters, runners, and metrics."""

from .adapters import case_to_prediction, igblast_airr_to_prediction, normalize_call_set
from .audit import audit_criteria_report
from .comparison import (
    COMPARISON_POLICY_TEMPLATES,
    DEFAULT_COMPARISON_METRICS,
    MULTIPLE_COMPARISON_CORRECTIONS,
    build_model_comparison_report,
    comparison_policy_catalog,
    validate_comparison_policy_catalog,
)
from .contract import (
    PredictionValidationAccumulator,
    prediction_contract,
    validate_prediction,
    validate_predictions,
    validate_predictions_for_cases,
)
from .context import case_contexts
from .diagnostics import (
    AlleleCallingDiagnosticsAccumulator,
    BoundaryDiagnosticsAccumulator,
    build_allele_calling_diagnostics,
    build_boundary_diagnostics,
)
from .matching import PredictionMatchResult, align_predictions_to_cases
from .scoring import (
    audit_scoring_runtime,
    compact_summary,
    score_cases,
    score_one_case,
    scoring_manifest_catalog,
    validate_scoring_manifest,
)
from .model_adapters import dnalignair_predictor
from .online import OnlineBenchmarkReport, run_online_benchmark
from .performance import (
    PERFORMANCE_GLOBAL_KEYS,
    PERFORMANCE_PREDICTION_FIELD_KEYS,
    PerformanceAccumulator,
    annotate_predictions_with_performance,
    current_rss_mb,
    normalize_performance_summary,
    performance_metrics_from_summary,
    prediction_performance_metrics,
    profile_predictor_call,
    summarize_prediction_performance,
)
from .report import build_assay_report
from .report_validation import validate_benchmark_report_contract
from .runner import build_benchmark_report, run_benchmark, run_benchmark_report
from .uncertainty import DEFAULT_BOOTSTRAP_METRICS, bootstrap_metric_intervals

__all__ = [
    "DEFAULT_BOOTSTRAP_METRICS",
    "COMPARISON_POLICY_TEMPLATES",
    "DEFAULT_COMPARISON_METRICS",
    "MULTIPLE_COMPARISON_CORRECTIONS",
    "OnlineBenchmarkReport",
    "AlleleCallingDiagnosticsAccumulator",
    "BoundaryDiagnosticsAccumulator",
    "PredictionMatchResult",
    "PredictionValidationAccumulator",
    "PERFORMANCE_GLOBAL_KEYS",
    "PERFORMANCE_PREDICTION_FIELD_KEYS",
    "PerformanceAccumulator",
    "align_predictions_to_cases",
    "annotate_predictions_with_performance",
    "audit_criteria_report",
    "audit_scoring_runtime",
    "build_benchmark_report",
    "build_allele_calling_diagnostics",
    "build_boundary_diagnostics",
    "build_assay_report",
    "build_model_comparison_report",
    "bootstrap_metric_intervals",
    "case_contexts",
    "case_to_prediction",
    "compact_summary",
    "comparison_policy_catalog",
    "validate_comparison_policy_catalog",
    "current_rss_mb",
    "dnalignair_predictor",
    "igblast_airr_to_prediction",
    "normalize_call_set",
    "normalize_performance_summary",
    "performance_metrics_from_summary",
    "prediction_performance_metrics",
    "prediction_contract",
    "profile_predictor_call",
    "run_online_benchmark",
    "run_benchmark",
    "run_benchmark_report",
    "score_cases",
    "score_one_case",
    "scoring_manifest_catalog",
    "summarize_prediction_performance",
    "validate_scoring_manifest",
    "validate_prediction",
    "validate_benchmark_report_contract",
    "validate_predictions",
    "validate_predictions_for_cases",
]
