from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .system import current_rss_mb
from .metrics import (
    performance_metrics_from_summary,
    normalize_performance_summary,
    _as_float,
)

Predictor = Callable[[list[str]], list[dict[str, Any] | None]]


def annotate_predictions_with_performance(
    predictions: Iterable[dict[str, Any] | None],
    summary: Mapping[str, Any],
) -> list[dict[str, Any] | None]:
    """Attach per-read profiler metrics to predictions without overwriting tool-provided values."""

    metrics = performance_metrics_from_summary(summary)
    annotated: list[dict[str, Any] | None] = []
    for pred in predictions:
        if pred is None:
            annotated.append(None)
            continue
        row = dict(pred)
        if "seconds_per_read" in metrics:
            row.setdefault("runtime_seconds", metrics["seconds_per_read"])
            row.setdefault("seconds_per_read", metrics["seconds_per_read"])
        if "milliseconds_per_read" in metrics:
            row.setdefault("runtime_ms", metrics["milliseconds_per_read"])
            row.setdefault("milliseconds_per_read", metrics["milliseconds_per_read"])
        if "reads_per_second" in metrics:
            row.setdefault("reads_per_second", metrics["reads_per_second"])
        for key in ("peak_memory_mb", "peak_memory_delta_mb", "python_tracemalloc_peak_mb"):
            if key in metrics:
                row.setdefault(key, metrics[key])
        row.setdefault("benchmark_profile_source", summary.get("source", "python_predictor_profile"))
        if "wall_time_seconds" in summary:
            row.setdefault("benchmark_wall_time_seconds", summary["wall_time_seconds"])
        if "n_sequences" in summary:
            row.setdefault("benchmark_batch_size", summary["n_sequences"])
        annotated.append(row)
    return annotated


def profile_predictor_call(
    predictor: Predictor,
    reads: Iterable[str],
    *,
    profile_memory: bool = True,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    """Run a predictor call and return predictions plus runtime/memory metadata."""

    batch = list(reads)
    rss_before = current_rss_mb() if profile_memory else None
    was_tracing = tracemalloc.is_tracing()
    started_tracing = False
    if profile_memory:
        if not was_tracing:
            tracemalloc.start()
            started_tracing = True
        else:
            try:
                tracemalloc.reset_peak()
            except AttributeError:  # pragma: no cover - old Python fallback.
                pass
    start = time.perf_counter()
    try:
        predictions = predictor(batch)
    finally:
        elapsed = time.perf_counter() - start
        trace_peak_mb = None
        if profile_memory and tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            trace_peak_mb = peak / (1024.0 * 1024.0)
            if started_tracing:
                tracemalloc.stop()
    rss_after = current_rss_mb() if profile_memory else None
    raw_summary: dict[str, Any] = {
        "source": "python_predictor_profile",
        "n_sequences": len(batch),
        "wall_time_seconds": elapsed,
    }
    if rss_before is not None:
        raw_summary["rss_max_mb_before"] = rss_before
    if rss_after is not None:
        raw_summary["rss_max_mb_after"] = rss_after
        raw_summary["peak_memory_mb"] = rss_after
    if rss_before is not None and rss_after is not None:
        raw_summary["peak_memory_delta_mb"] = max(0.0, rss_after - rss_before)
    if trace_peak_mb is not None:
        raw_summary["python_tracemalloc_peak_mb"] = trace_peak_mb
        if raw_summary.get("peak_memory_mb") is None:
            raw_summary["peak_memory_mb"] = trace_peak_mb
    summary = normalize_performance_summary(raw_summary, n_sequences=len(batch)) or raw_summary
    return annotate_predictions_with_performance(predictions, summary), summary


class PerformanceAccumulator:
    """Combine per-batch performance profiles into one benchmark-level summary."""

    def __init__(self, *, source: str = "python_predictor_profile") -> None:
        self.source = source
        self.n_batches = 0
        self.n_sequences = 0
        self.wall_time_seconds = 0.0
        self.max_values: dict[str, float] = {}

    def update(self, summary: Mapping[str, Any] | None) -> None:
        normalized = normalize_performance_summary(summary)
        if not normalized:
            return
        self.n_batches += 1
        self.n_sequences += int(normalized.get("n_sequences") or 0)
        wall_time = _as_float(normalized.get("wall_time_seconds"))
        if wall_time is not None:
            self.wall_time_seconds += wall_time
        for key in (
            "peak_memory_mb",
            "peak_memory_delta_mb",
            "python_tracemalloc_peak_mb",
            "rss_max_mb_before",
            "rss_max_mb_after",
        ):
            value = _as_float(normalized.get(key))
            if value is not None:
                self.max_values[key] = max(self.max_values.get(key, value), value)

    def to_dict(self) -> dict[str, Any] | None:
        if self.n_batches == 0:
            return None
        summary: dict[str, Any] = {
            "source": self.source,
            "n_batches": self.n_batches,
            "n_sequences": self.n_sequences,
            "wall_time_seconds": self.wall_time_seconds,
            **self.max_values,
        }
        return normalize_performance_summary(summary, n_sequences=self.n_sequences)
