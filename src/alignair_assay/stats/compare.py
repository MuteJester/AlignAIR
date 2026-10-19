"""Paired two-tool comparison: analytic paired tests, Holm-corrected verdicts."""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from statistics import NormalDist

from alignair_assay.metrics import CaseScores
from alignair_assay.stats.aggregate import summarize

# Below this many paired cases a metric is never declared different (p = 1).
MIN_PAIRED = 10


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down: reject the k-th smallest p while p <= alpha / (m - k); stop at the first failure.

    Ties in p are ordered by metric name so the outcome is deterministic.
    """
    m = len(pvalues)
    out = {name: False for name in pvalues}
    for k, (name, p) in enumerate(sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))):
        if p > alpha / (m - k):
            break
        out[name] = True
    return out


def _finite(x) -> bool:
    return x is not None and math.isfinite(x)


def mcnemar_exact(n_b_only: int, n_a_only: int) -> float:
    """Exact two-sided McNemar p-value: a binomial(n, 1/2) test on the discordant pairs (1.0 when there are none)."""
    n = n_b_only + n_a_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(n_b_only, n_a_only) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def paired_z(diffs: list[float]) -> float:
    """Two-sided p-value of a paired z-test on the differences (mean / (sd / sqrt(n)), normal tail)."""
    n = len(diffs)
    mean = math.fsum(diffs) / n
    var = math.fsum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    return math.erfc(abs(mean) / math.sqrt(var / n) / math.sqrt(2.0))


def compare(a: dict[str, CaseScores], b: dict[str, CaseScores], metrics: list[str], *, alpha: float = 0.05,
            lower_is_better: set[str]) -> dict[str, dict]:
    """Paired comparison of tool ``a`` against tool ``b`` for each metric in ``metrics``.

    A metric's paired subset is the cases where both tools have a finite value. ``a``/``b`` summarize each tool
    over all its own cases (with coverage); ``mean_a_paired``/``mean_b_paired`` are the means on the paired subset
    and ``delta`` is their difference (b - a).

    p-values are analytic: an exact two-sided McNemar test when every paired value is 0 or 1, otherwise a
    two-sided paired z-test on the differences, and 1.0 below ``MIN_PAIRED`` pairs. Verdicts are Holm-corrected
    across ``metrics`` at ``alpha``, oriented by ``lower_is_better``. ``ci95`` is delta +/- 1.96 standard errors;
    ``ci`` is the simultaneous Bonferroni interval (z at 1 - alpha / (2m)), conservative relative to the Holm
    decisions. Both are None below two pairs.
    """
    sum_a = summarize(a, list(a))
    sum_b = summarize(b, list(b))
    m = len(metrics)
    normal = NormalDist()
    z95 = normal.inv_cdf(0.975)
    z_family = normal.inv_cdf(1.0 - alpha / (2 * m)) if m else z95
    res: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    for name in metrics:
        sign = -1.0 if name in lower_is_better else 1.0
        pairs = [(a[c][name], b[c][name]) for c in a
                 if c in b and _finite(a[c].get(name)) and _finite(b[c].get(name))]
        n = len(pairs)
        raw = [y - x for x, y in pairs]
        d = [sign * r for r in raw]
        sa, sb = sum_a.get(name), sum_b.get(name)
        binary = n > 0 and all(x in (0.0, 1.0) and y in (0.0, 1.0) for x, y in pairs)
        entry = {"a": sa, "b": sb, "n_paired": n,
                 "coverage_a": sa.coverage if sa else 0.0, "coverage_b": sb.coverage if sb else 0.0,
                 "mean_a_paired": math.fsum(x for x, _ in pairs) / n if n else None,
                 "mean_b_paired": math.fsum(y for _, y in pairs) / n if n else None,
                 "delta": math.fsum(raw) / n if n else None,
                 "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d), "ties": sum(v == 0 for v in d),
                 "ci": None, "ci95": None}
        if n >= 2:
            delta = entry["delta"]
            se = math.sqrt(math.fsum((r - delta) ** 2 for r in raw) / (n - 1) / n)
            entry["ci95"] = (delta - z95 * se, delta + z95 * se)
            entry["ci"] = (delta - z_family * se, delta + z_family * se)
        if n < MIN_PAIRED:
            entry["test"], p = "too_few_pairs", 1.0
        elif binary:
            entry["test"], p = "mcnemar_exact", mcnemar_exact(sum(r > 0 for r in raw), sum(r < 0 for r in raw))
        else:
            entry["test"], p = "paired_z", paired_z(raw)
        entry["p_value"] = pvals[name] = p
        res[name] = entry
    reject = holm(pvals, alpha)
    for name, entry in res.items():
        direction = (entry["delta"] or 0.0) * (-1.0 if name in lower_is_better else 1.0)
        entry["verdict"] = ("b_better" if reject[name] and direction > 0
                            else "a_better" if reject[name] and direction < 0 else "tie_or_inconclusive")
    return res


def to_dict(obj):
    """JSON-ready copy of a summary or comparison result (dataclasses become dicts, tuples become lists)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_dict(asdict(obj))
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj
