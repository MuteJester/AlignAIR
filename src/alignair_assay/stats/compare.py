"""Paired two-tool comparison: analytic paired tests, cluster-aware when cases are grouped, Holm-corrected verdicts.

The reads of one clone, or the two strands of one read, are not independent: a test that treats them as
independent pairs overstates significance when a few clusters carry the difference. Given ``clusters``, ``compare``
tests the per-cluster sums of the paired differences instead (see ``compare``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from statistics import NormalDist

from alignair_assay.core.cases import evenly_spaced
from alignair_assay.metrics import CaseScores
from alignair_assay.stats.aggregate import summarize

# Below this many paired cases a metric is never declared different (p = 1).
MIN_PAIRED = 10
# Below this many clusters a clustered comparison is never declared different (p = 1).
MIN_CLUSTERS = 10


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


def cluster_score_z(sums: list[float]) -> float:
    """Two-sided p-value of the cluster score test on the per-cluster sums ``D_k`` of the paired differences:
    z = sum(D_k) / sqrt(sum(D_k^2)), p = erfc(|z| / sqrt(2)); 1.0 when every sum is 0. For 0/1 metrics it is the
    clustered McNemar test of Durkalski et al. (2003)."""
    squares = math.fsum(s * s for s in sums)
    if squares == 0.0:
        return 1.0
    return math.erfc(abs(math.fsum(sums)) / math.sqrt(squares) / math.sqrt(2.0))


def compare(a: dict[str, CaseScores], b: dict[str, CaseScores], metrics: list[str], *, alpha: float = 0.05,
            lower_is_better: set[str], clusters: Mapping[str, str] | None = None) -> dict[str, dict]:
    """Paired comparison of tool ``a`` against tool ``b`` for each metric in ``metrics``.

    A metric's paired subset is the cases where both tools have a finite value. ``a``/``b`` summarize each tool
    over all its own cases (with coverage); ``mean_a_paired``/``mean_b_paired`` are the means on the paired subset
    and ``delta`` is their difference (b - a). ``wins``, ``losses`` and ``ties`` count the paired cases.

    p-values are analytic: an exact two-sided McNemar test when every paired value is 0 or 1, otherwise a
    two-sided paired z-test on the differences, and 1.0 below ``MIN_PAIRED`` pairs. Verdicts are Holm-corrected
    across ``metrics`` at ``alpha``, oriented by ``lower_is_better``. ``ci95`` is delta +/- 1.96 standard errors;
    ``ci`` is the simultaneous Bonferroni interval (z at 1 - alpha / (2m)), conservative relative to the Holm
    decisions. Both are None below two pairs. When every paired value is 0 or 1 both are clipped to [-1, 1], the
    range a difference of two rates can take.

    ``clusters`` maps a case id to its cluster (a case it does not name is a cluster of its own); ``n_clusters`` is
    the number of clusters among a metric's pairs. When every cluster of a metric holds one pair, the metric is
    compared as above. Otherwise, with ``D_k`` the sum of the paired differences in cluster k, ``n_k`` its pairs,
    ``K`` the clusters and ``N`` the pairs: the test is the cluster score test (``cluster_score_z``;
    ``clustered_mcnemar`` for 0/1 metrics, ``clustered_z`` otherwise), with p = 1 below ``MIN_CLUSTERS`` clusters
    (``too_few_clusters``; ``too_few_pairs`` comes first), and the intervals use the cluster-robust (CR1) standard
    error sqrt(K / (K - 1) * sum((D_k - n_k * delta)^2)) / N, from two clusters. Holm, ``delta`` and the means are
    the same either way.
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
        cases = [c for c in a if c in b and _finite(a[c].get(name)) and _finite(b[c].get(name))]
        pairs = [(a[c][name], b[c][name]) for c in cases]
        n = len(pairs)
        raw = [y - x for x, y in pairs]
        d = [sign * r for r in raw]
        sa, sb = sum_a.get(name), sum_b.get(name)
        binary = n > 0 and all(x in (0.0, 1.0) and y in (0.0, 1.0) for x, y in pairs)
        sums = _cluster_sums(cases, raw, clusters) if clusters else None
        k = n if sums is None else len(sums)
        entry = {"a": sa, "b": sb, "n_paired": n, "n_clusters": k,
                 "coverage_a": sa.coverage if sa else 0.0, "coverage_b": sb.coverage if sb else 0.0,
                 "mean_a_paired": math.fsum(x for x, _ in pairs) / n if n else None,
                 "mean_b_paired": math.fsum(y for _, y in pairs) / n if n else None,
                 "delta": math.fsum(raw) / n if n else None,
                 "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d), "ties": sum(v == 0 for v in d),
                 "ci": None, "ci95": None}
        limit = 1.0 if binary else math.inf
        if k == n:  # every cluster a singleton: the read-level comparison
            if n >= 2:
                delta = entry["delta"]
                se = math.sqrt(math.fsum((r - delta) ** 2 for r in raw) / (n - 1) / n)
                entry["ci95"] = _interval(delta, z95 * se, limit)
                entry["ci"] = _interval(delta, z_family * se, limit)
            if n < MIN_PAIRED:
                entry["test"], p = "too_few_pairs", 1.0
            elif binary:
                entry["test"], p = "mcnemar_exact", mcnemar_exact(sum(r > 0 for r in raw), sum(r < 0 for r in raw))
            else:
                entry["test"], p = "paired_z", paired_z(raw)
        else:
            delta = entry["delta"]
            if k >= 2:
                spread = math.fsum((total - size * delta) ** 2 for total, size in sums)
                se = math.sqrt(k / (k - 1) * spread) / n
                entry["ci95"] = _interval(delta, z95 * se, limit)
                entry["ci"] = _interval(delta, z_family * se, limit)
            if n < MIN_PAIRED:
                entry["test"], p = "too_few_pairs", 1.0
            elif k < MIN_CLUSTERS:
                entry["test"], p = "too_few_clusters", 1.0
            else:
                entry["test"] = "clustered_mcnemar" if binary else "clustered_z"
                p = cluster_score_z([total for total, _ in sums])
        entry["p_value"] = pvals[name] = p
        res[name] = entry
    reject = holm(pvals, alpha)
    for name, entry in res.items():
        direction = (entry["delta"] or 0.0) * (-1.0 if name in lower_is_better else 1.0)
        entry["verdict"] = ("b_better" if reject[name] and direction > 0
                            else "a_better" if reject[name] and direction < 0 else "tie_or_inconclusive")
    return res


def read_changes(a: dict[str, CaseScores], b: dict[str, CaseScores], metrics: list[str], *,
                 lower_is_better: set[str], examples: int = 10, groups: Mapping[str, str] | None = None) -> dict:
    """The cases ``b`` fixes and breaks against ``a``, per 0/1 metric of ``metrics``: over a metric's paired cases
    (both values finite, as ``compare`` pairs them), a case is ``fixed`` when ``a`` is wrong and ``b`` right there
    (0 -> 1, or 1 -> 0 for a ``lower_is_better`` metric) and ``broken`` the other way. A metric with no paired case,
    or any paired value other than 0 and 1, has no entry. Per metric: ``n_paired``, the ``fixed`` and ``broken``
    counts and ``examples`` case ids of each (``sample``: evenly spaced over the sorted ids, since a suite lists its
    cases stratum by stratum); ``reads``: the cases fixed, and broken, on at least one metric, with ``examples`` of
    each and, with ``groups`` (case -> a label: its stratum, or its donor and protocol), the ``groups``' counts
    (label -> ``fixed``, ``broken``; a case without a label counts under ``"unlabelled"``); ``a_only`` and
    ``b_only``: how many cases only ``a``, or only ``b``, has (never paired, so a caller that compares the same reads
    must check both are 0), with ``examples`` of each (``a_only_examples``, ``b_only_examples``). JSON-ready."""
    a_only, b_only = sorted(case for case in a if case not in b), sorted(case for case in b if case not in a)
    found, fixed_any, broken_any = {}, set(), set()
    for name in metrics:
        pairs = [(case, a[case][name], b[case][name]) for case in a
                 if case in b and _finite(a[case].get(name)) and _finite(b[case].get(name))]
        if not pairs or not all(x in (0.0, 1.0) and y in (0.0, 1.0) for _, x, y in pairs):
            continue
        sign = -1.0 if name in lower_is_better else 1.0
        fixed = sorted(case for case, x, y in pairs if sign * (y - x) > 0)
        broken = sorted(case for case, x, y in pairs if sign * (y - x) < 0)
        found[name] = {"n_paired": len(pairs), "fixed": len(fixed), "broken": len(broken),
                       "fixed_examples": sample(fixed, examples), "broken_examples": sample(broken, examples)}
        fixed_any.update(fixed)
        broken_any.update(broken)
    reads = {"fixed": len(fixed_any), "broken": len(broken_any),
             "fixed_examples": sample(sorted(fixed_any), examples),
             "broken_examples": sample(sorted(broken_any), examples)}
    if groups is not None:
        counts: dict[str, dict[str, int]] = {}
        for kind, cases in (("fixed", fixed_any), ("broken", broken_any)):
            for case in cases:
                label = groups.get(case, "unlabelled")
                counts.setdefault(label, {"fixed": 0, "broken": 0})[kind] += 1
        reads["groups"] = dict(sorted(counts.items()))
    return {"metrics": found, "reads": reads, "a_only": len(a_only), "b_only": len(b_only),
            "a_only_examples": sample(a_only, examples), "b_only_examples": sample(b_only, examples)}


def sample(ids: list[str], n: int) -> list[str]:
    """``ids`` when there are at most ``n``, else ``n`` of them evenly spaced (``core.cases.evenly_spaced``)."""
    return list(ids) if len(ids) <= n else evenly_spaced(ids, n)


def _cluster_sums(cases: list[str], raw: list[float], clusters: Mapping[str, str]) -> list[tuple[float, int]]:
    """Per cluster of the paired ``cases`` (a case ``clusters`` does not name is its own), in first-seen order:
    the sum of its paired differences ``raw`` and its number of pairs."""
    groups: dict[object, list[float]] = {}
    for case, diff in zip(cases, raw):
        cluster = clusters.get(case)
        groups.setdefault(("case", case) if cluster is None else ("cluster", cluster), []).append(diff)
    return [(math.fsum(diffs), len(diffs)) for diffs in groups.values()]


def _interval(center: float, half_width: float, limit: float) -> tuple[float, float]:
    """``center`` +/- ``half_width``, clipped to [-limit, limit]."""
    return max(center - half_width, -limit), min(center + half_width, limit)


def to_dict(obj):
    """JSON-ready copy of a summary or comparison result (dataclasses become dicts, tuples become lists)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_dict(asdict(obj))
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj
