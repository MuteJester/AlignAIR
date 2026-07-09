"""Per-axis struggle attribution: bucket per-read correctness by the read's
GenAIRR truth difficulty (SHM, indels, noise, length) to name the bottleneck."""
from typing import Sequence

from .state import AxisStat


def bucket_axis(records: Sequence[dict], axis: str, edges: Sequence[float],
                metric_key: str) -> AxisStat:
    edges = list(edges)
    sums = [0.0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)
    for r in records:
        v = float(r.get(axis, 0.0))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            # last bin is inclusive of the top edge
            in_bin = lo < v <= hi if i > 0 else lo <= v <= hi
            if in_bin:
                sums[i] += float(r.get(metric_key, 0.0))
                counts[i] += 1
                break
    bins = []
    for i in range(len(edges) - 1):
        label = f"{edges[i]:g}-{edges[i + 1]:g}"
        mean = sums[i] / counts[i] if counts[i] else 0.0
        bins.append((label, mean, counts[i]))
    return AxisStat(axis=axis, bins=tuple(bins))


_AXES = (
    ("shm", "mutation_rate", [0.0, 0.05, 0.15, 1.0]),
    ("indel", "indel_count", [0.0, 0.5, 3.0, 100.0]),
    ("noise", "noise_count", [0.0, 0.5, 3.0, 100.0]),
    ("length", "length", [0.0, 100.0, 250.0, 10000.0]),
)


def axis_breakdown(records: Sequence[dict], metric_key: str = "correct") -> tuple:
    out = []
    for name, key, edges in _AXES:
        st = bucket_axis(records, axis=key, edges=edges, metric_key=metric_key)
        out.append(AxisStat(axis=name, bins=st.bins))
    return tuple(out)
