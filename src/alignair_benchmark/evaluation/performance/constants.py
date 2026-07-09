from __future__ import annotations

_SECONDS_PER_READ_KEYS = (
    "seconds_per_read",
    "seconds_per_sequence",
    "time_per_read_seconds",
    "time_per_sequence_seconds",
    "runtime_seconds",
    "runtime_s",
    "elapsed_seconds",
    "elapsed_s",
    "latency_seconds",
    "latency_s",
)
_MILLISECONDS_PER_READ_KEYS = (
    "milliseconds_per_read",
    "ms_per_read",
    "milliseconds_per_sequence",
    "ms_per_sequence",
    "time_per_read_ms",
    "time_per_sequence_ms",
    "runtime_ms",
    "elapsed_ms",
    "latency_ms",
)
_READS_PER_SECOND_KEYS = (
    "reads_per_second",
    "read_per_second",
    "seqs_per_second",
    "sequences_per_second",
    "throughput",
)
_PEAK_MEMORY_MB_KEYS = (
    "peak_memory_mb",
    "peak_rss_mb",
    "rss_peak_mb",
    "rss_max_mb",
    "rss_max_mb_after",
    "max_rss_mb",
    "memory_mb",
)
_PEAK_MEMORY_BYTES_KEYS = (
    "peak_memory_bytes",
    "peak_rss_bytes",
    "rss_peak_bytes",
    "memory_bytes",
)
_PEAK_MEMORY_DELTA_MB_KEYS = (
    "peak_memory_delta_mb",
    "rss_peak_delta_mb",
    "rss_max_delta_mb",
    "memory_delta_mb",
)
_PYTHON_PEAK_MB_KEYS = (
    "python_tracemalloc_peak_mb",
    "tracemalloc_peak_mb",
    "python_peak_memory_mb",
)
_CANDIDATE_COUNT_KEYS = ("candidate_count", "n_candidates", "num_candidates", "candidate_calls")
_RERANK_COUNT_KEYS = ("rerank_count", "n_reranked", "num_reranked", "rerank_candidates")

PERFORMANCE_GLOBAL_KEYS = (
    "reads_per_second",
    "seconds_per_read",
    "milliseconds_per_read",
    "peak_memory_mb",
    "peak_memory_delta_mb",
    "python_tracemalloc_peak_mb",
    "candidate_count",
    "rerank_count",
)

PERFORMANCE_PREDICTION_FIELD_KEYS = tuple(
    dict.fromkeys(
        _SECONDS_PER_READ_KEYS
        + _MILLISECONDS_PER_READ_KEYS
        + _READS_PER_SECOND_KEYS
        + _PEAK_MEMORY_MB_KEYS
        + _PEAK_MEMORY_BYTES_KEYS
        + _PEAK_MEMORY_DELTA_MB_KEYS
        + _PYTHON_PEAK_MB_KEYS
        + _CANDIDATE_COUNT_KEYS
        + _RERANK_COUNT_KEYS
    )
)
