"""Shared evaluation statistics, cohort reports, and the model benchmark API."""
from .reports import (boundary_error_report, classification_report, constant_median_mae,
                      macro_accuracy, majority_class_baseline)
from .statistics import (bootstrap_competence_ci, bootstrap_mean_ci, paired_diff_ci,
                         wilson_interval)

_BENCHMARK_EXPORTS = ("default_strata", "format_text", "generate_labeled", "run_benchmark", "score")
__all__ = [*_BENCHMARK_EXPORTS, "boundary_error_report", "classification_report", "constant_median_mae",
           "macro_accuracy", "majority_class_baseline", "bootstrap_competence_ci", "bootstrap_mean_ci",
           "paired_diff_ci", "wilson_interval"]


def __getattr__(name):
    # Pure report/statistic imports must not initialize the prediction API or simulator benchmark.
    if name in _BENCHMARK_EXPORTS:
        from . import benchmark
        value = getattr(benchmark, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
