"""Prediction adapters, runners, and metrics."""

from .adapters import case_to_prediction, igblast_airr_to_prediction, normalize_call_set
from .metrics import compact_summary, score_cases, score_one_case
from .model_adapters import dnalignair_predictor
from .online import OnlineBenchmarkReport, case_contexts, run_online_benchmark
from .report import build_assay_report
from .runner import run_benchmark

__all__ = [
    "OnlineBenchmarkReport",
    "build_assay_report",
    "case_contexts",
    "case_to_prediction",
    "compact_summary",
    "dnalignair_predictor",
    "igblast_airr_to_prediction",
    "normalize_call_set",
    "run_online_benchmark",
    "run_benchmark",
    "score_cases",
    "score_one_case",
]
