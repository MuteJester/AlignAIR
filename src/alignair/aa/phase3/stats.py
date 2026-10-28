"""Exact one-sided binomial (Clopper-Pearson) bounds + the Phase-3 coverage decision."""
from __future__ import annotations

from alignair.evaluate.statistics import clopper_pearson_interval as clopper_pearson


def coverage_decision(k: int, n: int, alpha: float, allow_support: bool) -> str:
    lo, hi = clopper_pearson(k, n, alpha)
    if hi < 0.95:
        return "failed"
    if allow_support and lo >= 0.95:
        return "supported"
    return "inconclusive"
