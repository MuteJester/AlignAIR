from __future__ import annotations

import math
from typing import Any, Mapping, Iterable

from .constants import (
    PERFORMANCE_GLOBAL_KEYS,
    _SECONDS_PER_READ_KEYS,
    _MILLISECONDS_PER_READ_KEYS,
    _READS_PER_SECOND_KEYS,
    _PEAK_MEMORY_MB_KEYS,
    _PEAK_MEMORY_BYTES_KEYS,
    _PEAK_MEMORY_DELTA_MB_KEYS,
    _PYTHON_PEAK_MB_KEYS,
    _CANDIDATE_COUNT_KEYS,
    _RERANK_COUNT_KEYS,
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple, set, dict)):
        return float(len(value))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping:
            value = _as_float(mapping[key])
            if value is not None:
                return value
    return None


def _positive(value: float | None) -> float | None:
    return value if value is not None and value > 0 else None


def prediction_performance_metrics(prediction: Mapping[str, Any] | None) -> dict[str, float]:
    """Extract per-sequence performance metrics from a prediction record.

    ``runtime_seconds``/``runtime_ms`` are interpreted as per-sequence latency in
    prediction rows. Total benchmark runtime belongs in report-level performance
    metadata and is normalized by ``normalize_performance_summary``.
    """

    if not prediction:
        return {}
    pred = prediction
    out: dict[str, float] = {}
    seconds = _first_float(pred, *_SECONDS_PER_READ_KEYS)
    milliseconds = _first_float(pred, *_MILLISECONDS_PER_READ_KEYS)
    reads_per_second = _first_float(pred, *_READS_PER_SECOND_KEYS)

    if seconds is None and milliseconds is not None:
        seconds = milliseconds / 1000.0
    if milliseconds is None and seconds is not None:
        milliseconds = seconds * 1000.0
    if reads_per_second is None and _positive(seconds) is not None:
        reads_per_second = 1.0 / seconds
    if seconds is None and _positive(reads_per_second) is not None:
        seconds = 1.0 / reads_per_second
        milliseconds = seconds * 1000.0
    if milliseconds is None and _positive(reads_per_second) is not None:
        milliseconds = 1000.0 / reads_per_second

    if seconds is not None:
        out["seconds_per_read"] = seconds
    if milliseconds is not None:
        out["milliseconds_per_read"] = milliseconds
    if reads_per_second is not None:
        out["reads_per_second"] = reads_per_second

    peak_memory_mb = _first_float(pred, *_PEAK_MEMORY_MB_KEYS)
    peak_memory_bytes = _first_float(pred, *_PEAK_MEMORY_BYTES_KEYS)
    if peak_memory_mb is None and peak_memory_bytes is not None:
        peak_memory_mb = peak_memory_bytes / (1024.0 * 1024.0)
    if peak_memory_mb is not None:
        out["peak_memory_mb"] = peak_memory_mb

    peak_memory_delta_mb = _first_float(pred, *_PEAK_MEMORY_DELTA_MB_KEYS)
    if peak_memory_delta_mb is not None:
        out["peak_memory_delta_mb"] = peak_memory_delta_mb

    python_peak = _first_float(pred, *_PYTHON_PEAK_MB_KEYS)
    if python_peak is not None:
        out["python_tracemalloc_peak_mb"] = python_peak

    candidate_count = _first_float(pred, *_CANDIDATE_COUNT_KEYS)
    if candidate_count is not None:
        out["candidate_count"] = candidate_count

    rerank_count = _first_float(pred, *_RERANK_COUNT_KEYS)
    if rerank_count is not None:
        out["rerank_count"] = rerank_count

    return out


def normalize_performance_summary(
    summary: Mapping[str, Any] | None,
    *,
    n_sequences: int | None = None,
    source: str | None = None,
) -> dict[str, Any] | None:
    """Normalize flexible runtime/memory metadata into benchmark report fields."""

    if not summary:
        return None
    data = dict(summary)
    resolved_n = _first_float(
        data,
        "n_sequences",
        "n_reads",
        "n_cases",
        "sequence_count",
        "read_count",
    )
    if n_sequences is not None:
        resolved_n = float(n_sequences)
    resolved_n_int = int(resolved_n) if resolved_n is not None and resolved_n >= 0 else None

    wall_time = _first_float(
        data,
        "wall_time_seconds",
        "elapsed_seconds",
        "total_runtime_seconds",
        "runtime_seconds",
        "runtime_s",
    )
    wall_time_ms = _first_float(
        data,
        "wall_time_ms",
        "elapsed_ms",
        "total_runtime_ms",
        "runtime_ms",
    )
    if wall_time is None and wall_time_ms is not None:
        wall_time = wall_time_ms / 1000.0

    seconds = _first_float(
        data,
        "seconds_per_read",
        "seconds_per_sequence",
        "time_per_read_seconds",
        "time_per_sequence_seconds",
    )
    milliseconds = _first_float(
        data,
        "milliseconds_per_read",
        "ms_per_read",
        "milliseconds_per_sequence",
        "ms_per_sequence",
        "time_per_read_ms",
        "time_per_sequence_ms",
    )
    reads_per_second = _first_float(data, *_READS_PER_SECOND_KEYS)

    if seconds is None and milliseconds is not None:
        seconds = milliseconds / 1000.0
    if seconds is None and wall_time is not None and resolved_n_int:
        seconds = wall_time / resolved_n_int
    if milliseconds is None and seconds is not None:
        milliseconds = seconds * 1000.0
    if reads_per_second is None and wall_time is not None and wall_time > 0 and resolved_n_int is not None:
        reads_per_second = resolved_n_int / wall_time
    if reads_per_second is None and _positive(seconds) is not None:
        reads_per_second = 1.0 / seconds
    if seconds is None and _positive(reads_per_second) is not None:
        seconds = 1.0 / reads_per_second
        milliseconds = seconds * 1000.0
    if wall_time is None and seconds is not None and resolved_n_int is not None:
        wall_time = seconds * resolved_n_int

    peak_memory_mb = _first_float(data, *_PEAK_MEMORY_MB_KEYS)
    peak_memory_bytes = _first_float(data, *_PEAK_MEMORY_BYTES_KEYS)
    if peak_memory_mb is None and peak_memory_bytes is not None:
        peak_memory_mb = peak_memory_bytes / (1024.0 * 1024.0)
    peak_memory_delta_mb = _first_float(data, *_PEAK_MEMORY_DELTA_MB_KEYS)
    python_peak = _first_float(data, *_PYTHON_PEAK_MB_KEYS)

    out: dict[str, Any] = {}
    if source or data.get("source"):
        out["source"] = source or data.get("source")
    if "n_batches" in data:
        n_batches = _first_float(data, "n_batches")
        if n_batches is not None:
            out["n_batches"] = int(n_batches)
    if resolved_n_int is not None:
        out["n_sequences"] = resolved_n_int
    if wall_time is not None:
        out["wall_time_seconds"] = wall_time
    if seconds is not None:
        out["seconds_per_read"] = seconds
    if milliseconds is not None:
        out["milliseconds_per_read"] = milliseconds
    if reads_per_second is not None:
        out["reads_per_second"] = reads_per_second
    if peak_memory_mb is not None:
        out["peak_memory_mb"] = peak_memory_mb
    if peak_memory_delta_mb is not None:
        out["peak_memory_delta_mb"] = peak_memory_delta_mb
    if python_peak is not None:
        out["python_tracemalloc_peak_mb"] = python_peak
    for key in ("rss_max_mb_before", "rss_max_mb_after", "candidate_count", "rerank_count"):
        value = _first_float(data, key)
        if value is not None:
            out[key] = value
    return out if any(key in out for key in PERFORMANCE_GLOBAL_KEYS) or "wall_time_seconds" in out else None


def performance_metrics_from_summary(summary: Mapping[str, Any] | None) -> dict[str, float]:
    """Return report-level metrics that should also appear in ``results.overall.global``."""

    if not summary:
        return {}
    out: dict[str, float] = {}
    for key in PERFORMANCE_GLOBAL_KEYS:
        value = _as_float(summary.get(key))
        if value is not None:
            out[key] = value
    return out


def summarize_prediction_performance(
    predictions: Iterable[Mapping[str, Any] | None],
    *,
    n_sequences: int | None = None,
) -> dict[str, Any] | None:
    """Summarize performance values embedded in saved prediction rows."""

    metric_values: dict[str, list[float]] = {key: [] for key in PERFORMANCE_GLOBAL_KEYS}
    count = 0
    for pred in predictions:
        count += 1
        for key, value in prediction_performance_metrics(pred).items():
            metric_values[key].append(value)
    observed = {key: values for key, values in metric_values.items() if values}
    if not observed:
        return None
    resolved_n = n_sequences if n_sequences is not None else count
    summary: dict[str, Any] = {
        "source": "prediction_fields",
        "n_sequences": resolved_n,
    }
    for key in ("seconds_per_read", "milliseconds_per_read", "reads_per_second", "candidate_count", "rerank_count"):
        values = observed.get(key)
        if values:
            summary[key] = sum(values) / len(values)
    for key in ("peak_memory_mb", "peak_memory_delta_mb", "python_tracemalloc_peak_mb"):
        values = observed.get(key)
        if values:
            summary[key] = max(values)
    if "seconds_per_read" in summary:
        summary["wall_time_seconds"] = summary["seconds_per_read"] * resolved_n
    return normalize_performance_summary(summary, n_sequences=resolved_n)
