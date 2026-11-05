"""The model benchmark API (``benchmark``, imported on first use)."""
_BENCHMARK_EXPORTS = ("default_strata", "format_text", "generate_labeled", "run_benchmark", "score")
__all__ = [*_BENCHMARK_EXPORTS]


def __getattr__(name):
    # Importing the package must not initialize the prediction API or the simulator benchmark.
    if name in _BENCHMARK_EXPORTS:
        from . import benchmark
        value = getattr(benchmark, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
