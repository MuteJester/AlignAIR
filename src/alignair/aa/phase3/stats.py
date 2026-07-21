"""Exact one-sided binomial (Clopper-Pearson) bounds + the Phase-3 coverage decision."""
from __future__ import annotations

from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))
    return lo, hi


def coverage_decision(k: int, n: int, alpha: float, allow_support: bool) -> str:
    lo, hi = clopper_pearson(k, n, alpha)
    if hi < 0.95:
        return "failed"
    if allow_support and lo >= 0.95:
        return "supported"
    return "inconclusive"
