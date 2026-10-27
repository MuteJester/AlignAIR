"""Shared interval estimators with explicit, reproducible resampling conventions."""
from __future__ import annotations

import math
from operator import index
import random

import numpy as np

from ..errors import ConfigurationError, InputError

DEFAULT_MAX_RESAMPLE_ELEMENTS = 1_000_000


def _integer(value, field, minimum):
    try:
        if isinstance(value, bool) or index(value) < minimum:
            raise ValueError
        return index(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be an integer >= {minimum}", context={"field": field}) from exc


def _finite_values(values, field):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InputError("observations must be a finite numeric vector", context={"field": field}) from exc
    if result.ndim != 1 or not np.isfinite(result).all():
        raise InputError("observations must be a finite numeric vector", context={"field": field})
    return result


def _resampling_options(n_boot, seed, alpha):
    n_boot = _integer(n_boot, "n_boot", 1)
    seed = _integer(seed, "seed", 0)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ConfigurationError("alpha must be between zero and one", context={"field": "alpha"})
    return n_boot, seed


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Descriptive binomial interval; an empty observation set retains the historical zero pair."""
    k, n = _integer(k, "k", 0), _integer(n, "n", 0)
    if k > n:
        raise InputError("successes cannot exceed the observation count", context={"k": k, "n": n})
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0:
        raise ConfigurationError("z must be positive and finite", context={"field": "z"})
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 0, *, alpha: float = 0.05,
                      max_resample_elements: int = DEFAULT_MAX_RESAMPLE_ELEMENTS) -> tuple[float, float]:
    """NumPy percentile interval for an IID per-observation mean.

    Draws use a local default_rng and linear percentile interpolation. Resamples are processed
    in chunks without changing the random draw order. The element budget bounds each index/value
    matrix, with a minimum of one whole resample; it is not a bound on the input vector or output
    means. Empty observations retain the historical (0, 0) result, not evidence of certainty.
    """
    n_boot, seed = _resampling_options(n_boot, seed, alpha)
    budget = _integer(max_resample_elements, "max_resample_elements", 1)
    values = _finite_values(values, "values")
    if not values.size:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    rows = max(1, budget // values.size)
    for start in range(0, n_boot, rows):
        stop = min(start + rows, n_boot)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return (float(np.percentile(means, alpha / 2 * 100)),
            float(np.percentile(means, (1 - alpha / 2) * 100)))


def paired_diff_ci(diffs, n_boot: int = 2000, seed: int = 0, *,
                   max_resample_elements: int = DEFAULT_MAX_RESAMPLE_ELEMENTS) -> dict:
    """Interval for paired per-observation advantages; positive values favor the candidate.

    Inputs must already be differences on the same aligned observations. This is not a cluster-
    or class-resampling procedure. Bounds use a fixed 95% interval, matching the ci95 result key.
    """
    values = _finite_values(diffs, "diffs")
    lo, hi = bootstrap_mean_ci(values, n_boot, seed,
                               max_resample_elements=max_resample_elements)
    mean = float(sum(values) / len(values)) if len(values) else 0.0
    return {"mean": mean, "ci95": (lo, hi), "lower_gt_zero": lo > 0.0}


def bootstrap_competence_ci(values, n_boot: int = 1000, alpha: float = 0.05,
                            seed: int = 0) -> tuple[float, float, float]:
    """Historical gym estimator: Python Random draws and sorted order-statistic endpoints.

    Kept distinct from NumPy's linearly interpolated estimator so existing competence/pacing
    results retain their seeded convention. Returns (mean, lower, upper); empty input is zeros.
    Storage is one input vector plus n_boot means, without a resample matrix.
    """
    n_boot, seed = _resampling_options(n_boot, seed, alpha)
    vals = _finite_values(values, "values").tolist()
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(vals) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (mean, lo, hi)
