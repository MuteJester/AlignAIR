"""Prediction adapters, runners, and metrics."""

from .adapters import case_to_prediction, igblast_airr_to_prediction, normalize_call_set
from .audit import audit_criteria_report
from .contract import (
    PredictionValidationAccumulator,
    prediction_contract,
    validate_prediction,
    validate_predictions,
)
from .context import case_contexts
from .diagnostics import (
    AlleleCallingDiagnosticsAccumulator,
    BoundaryDiagnosticsAccumulator,
    build_allele_calling_diagnostics,
    build_boundary_diagnostics,
)
from .matching import PredictionMatchResult, align_predictions_to_cases
from .metrics import compact_summary, score_cases, score_one_case
from .model_adapters import dnalignair_predictor
from .online import OnlineBenchmarkReport, run_online_benchmark
from .report import build_assay_report
from .runner import build_benchmark_report, run_benchmark, run_benchmark_report
from .uncertainty import DEFAULT_BOOTSTRAP_METRICS, bootstrap_metric_intervals

__all__ = [
    "DEFAULT_BOOTSTRAP_METRICS",
    "OnlineBenchmarkReport",
    "AlleleCallingDiagnosticsAccumulator",
    "BoundaryDiagnosticsAccumulator",
    "PredictionMatchResult",
    "PredictionValidationAccumulator",
    "align_predictions_to_cases",
    "audit_criteria_report",
    "build_benchmark_report",
    "build_allele_calling_diagnostics",
    "build_boundary_diagnostics",
    "build_assay_report",
    "bootstrap_metric_intervals",
    "case_contexts",
    "case_to_prediction",
    "compact_summary",
    "dnalignair_predictor",
    "igblast_airr_to_prediction",
    "normalize_call_set",
    "prediction_contract",
    "run_online_benchmark",
    "run_benchmark",
    "run_benchmark_report",
    "score_cases",
    "score_one_case",
    "validate_prediction",
    "validate_predictions",
]
