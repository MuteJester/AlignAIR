"""Paired two-tool comparison: bootstrap over cases, Holm-corrected verdicts."""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass

import numpy as np

from alignair_assay.metrics import CaseScores
from alignair_assay.stats.aggregate import summarize


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


def compare(a: dict[str, CaseScores], b: dict[str, CaseScores], metrics: list[str], *, n_boot: int = 1000,
            seed: int = 123, alpha: float = 0.05, lower_is_better: set[str]) -> dict[str, dict]:
    """Paired comparison of tool ``a`` against tool ``b`` for each metric in ``metrics``.

    ``delta`` is the mean raw difference (b - a) over cases where both tools have a finite value. Verdicts use a
    two-sided paired-bootstrap p-value on the direction-adjusted difference, Holm-corrected across ``metrics``.
    ``ci`` is the percentile interval of the bootstrap mean at the familywise Bonferroni level 1 - alpha / m
    (simultaneous, hence conservative relative to the Holm decisions); ``ci95`` is the plain 95% interval.
    Both are None when no case is paired.
    """
    sum_a = summarize(a, list(a))
    sum_b = summarize(b, list(b))
    m = len(metrics)
    rng = np.random.default_rng(seed)
    res: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    mean_d: dict[str, float] = {}
    for name in metrics:
        sign = -1.0 if name in lower_is_better else 1.0
        raw = [b[c][name] - a[c][name] for c in a
               if c in b and _finite(a[c].get(name)) and _finite(b[c].get(name))]
        raw_arr = np.asarray(raw, dtype=float)
        d = raw_arr * sign
        n = len(d)
        sa, sb = sum_a.get(name), sum_b.get(name)
        entry = {"a": sa, "b": sb, "n_paired": n,
                 "coverage_a": sa.coverage if sa else 0.0, "coverage_b": sb.coverage if sb else 0.0,
                 "delta": float(raw_arr.mean()) if n else None,
                 "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                 "ci": None, "ci95": None}
        p = 1.0
        if n:
            idx = rng.integers(0, n, size=(n_boot, n))
            boot_raw = raw_arr[idx].mean(axis=1)
            boot_d = boot_raw * sign
            if np.any(d != 0):
                p = min(1.0, 2.0 * min(float((boot_d <= 0).mean()), float((boot_d >= 0).mean())))
            tail = 100.0 * alpha / m / 2.0
            entry["ci"] = (float(np.percentile(boot_raw, tail)), float(np.percentile(boot_raw, 100.0 - tail)))
            entry["ci95"] = (float(np.percentile(boot_raw, 2.5)), float(np.percentile(boot_raw, 97.5)))
            mean_d[name] = float(d.mean())
        entry["p_boot"] = p
        pvals[name] = p
        res[name] = entry
    reject = holm(pvals, alpha)
    for name, entry in res.items():
        md = mean_d.get(name, 0.0)
        entry["verdict"] = ("b_better" if reject[name] and md > 0
                            else "a_better" if reject[name] and md < 0 else "tie_or_inconclusive")
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
