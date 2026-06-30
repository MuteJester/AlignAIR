from __future__ import annotations

import math
from typing import Sequence, Tuple

# one calibration row per read/gene: (candidate scores, indices of truth alleles present in C)
Row = Tuple[Sequence[float], Sequence[int]]


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def multipos_nll(rows: Sequence[Row], T: float) -> float:
    """Mean multi-positive NLL: -log( sum_{c in truth} e^{s_c/T} / sum_{c} e^{s_c/T} ).
    Rows whose truth set is absent from the candidate list are skipped (top-k miss — a
    candidate-recall failure that no threshold can fix)."""
    total, n = 0.0, 0
    for scores, pos in rows:
        if not pos or not scores:
            continue
        s = [x / T for x in scores]
        denom = _logsumexp(s)
        numer = _logsumexp([s[j] for j in pos])
        total += -(numer - denom)
        n += 1
    return total / max(n, 1)


def fit_temperature(rows: Sequence[Row], grid: Sequence[float] | None = None) -> float:
    """Grid-search the per-gene temperature that minimizes multi-positive NLL."""
    grid = grid or [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    best_T, best = 1.0, float("inf")
    for T in grid:
        v = multipos_nll(rows, T)
        if v < best:
            best, best_T = v, T
    return best_T
