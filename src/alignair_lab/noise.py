"""Run-to-run training noise, and the rule that decides whether a screen's Δ is beyond it. Pure: the assay's comparison
summaries in (``evaluate(..., against=...)``'s ``result.against["summary"]``), numbers out.

Two trainings of one architecture on the same records differ: GPU kernels are not deterministic, and replicate runs
draw other initial weights. A candidate's Δ against a baseline screen holds that noise beside the candidate's own
effect, and the assay's CIs, which resample the reads of two fixed checkpoints, leave it out. ``calibrate`` measures
it with baseline screens of several init seeds, compared pair by pair. Per metric:

- ``pooled``: a comparison's Δ (candidate − baseline) pooled over its suites, each weighted by its paired reads.
- ``null_band``: the largest |pooled Δ| of the pairs of baseline screens.
- ``decide``: a candidate's pooled Δ is ``beyond noise`` when it exceeds the null band and at least 2/3 of the suites
  have a Δ of its sign, counting the suites with enough clusters for the assay's verdict on the metric (its test is not
  ``too_few_clusters``: fewer than ``stats.compare.MIN_CLUSTERS``), or every suite that compares it when none has
  them (the ``basis``: ``clusters`` or ``all``); otherwise ``within noise``; ``uncalibrated`` without a band for the
  metric, and ``uncalibrated: band 0`` when its band is 0: baseline screens that never differed on it bound nothing.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

BEYOND, WITHIN, UNCALIBRATED = "beyond noise", "within noise", "uncalibrated"
ZERO_BAND = "uncalibrated: band 0"
FEW_CLUSTERS = "too_few_clusters"         # the assay's test when a metric's pairs fall in too few clusters for a verdict


def suite_deltas(comparison: Mapping, metric: str) -> dict[str, dict]:
    """Each suite that compares ``metric`` on paired reads -> its ``delta``, ``n_paired`` and ``clustered``: whether
    its pairs fall in enough clusters for the assay's verdict (its test is not ``too_few_clusters``)."""
    found = {}
    for name, suite in comparison["suites"].items():
        entry = (suite.get("headline") or {}).get(metric)
        if entry and entry.get("delta") is not None and entry.get("n_paired"):
            found[name] = {"delta": entry["delta"], "n_paired": entry["n_paired"],
                           "clustered": entry.get("test") != FEW_CLUSTERS}
    return found


def pooled(comparison: Mapping, metric: str) -> float | None:
    """``metric``'s Δ over the suites that compare it, weighted by their paired reads; None when no suite does."""
    suites = suite_deltas(comparison, metric)
    reads = sum(suite["n_paired"] for suite in suites.values())
    return sum(suite["delta"] * suite["n_paired"] for suite in suites.values()) / reads if reads else None


def _metrics(comparison: Mapping) -> list[str]:
    return list(dict.fromkeys(metric for suite in comparison["suites"].values()
                              for metric in suite.get("headline") or {}))


def pair(comparison: Mapping) -> dict:
    """One comparison of two baseline screens, as the null band keeps it: each metric's ``pooled`` Δ and each suite's
    Δs (``suites``), for the metrics some suite compares on paired reads."""
    metrics = [metric for metric in _metrics(comparison) if pooled(comparison, metric) is not None]
    suites = {}
    for metric in metrics:
        for name, suite in suite_deltas(comparison, metric).items():
            suites.setdefault(name, {})[metric] = suite["delta"]
    return {"pooled": {metric: pooled(comparison, metric) for metric in metrics}, "suites": suites}


def null_band(pairs: Sequence[Mapping]) -> dict[str, float]:
    """Each metric's null band: the largest |pooled Δ| among the ``pairs`` (``pair``) that compare it."""
    band: dict[str, float] = {}
    for found in pairs:
        for metric, delta in found["pooled"].items():
            band[metric] = max(band.get(metric, 0.), abs(delta))
    return band


def _scaled(found: Mapping, sign: int) -> dict:
    return {"pooled": {metric: sign * value for metric, value in found["pooled"].items()},
            "suites": {suite: {metric: sign * value for metric, value in values.items()}
                       for suite, values in found["suites"].items()}}


def seed_values(pairs: Sequence[Mapping], seeds: Sequence[int], *, relative_to: int) -> dict[str, dict]:
    """Each of a calibration's ``seeds`` (as a string, in order) -> its ``pooled`` value per metric and each suite's
    (``suites``: suite -> metric -> value), relative to the seed ``relative_to``: the Δs of the pair
    ``(relative_to, seed)`` of ``pairs`` (``{"a", "b", **pair(...)}``, b's Δs against a's), negated when the pair is
    ``(seed, relative_to)``. ``relative_to``'s own values are 0, on every metric and suite the others have. ValueError
    when no pair compares a seed with ``relative_to``."""
    found = {(entry["a"], entry["b"]): entry for entry in pairs}
    values = {}
    for seed in seeds:
        if seed == relative_to:
            continue
        if (relative_to, seed) in found:
            values[str(seed)] = _scaled(found[(relative_to, seed)], 1)
        elif (seed, relative_to) in found:
            values[str(seed)] = _scaled(found[(seed, relative_to)], -1)
        else:
            raise ValueError(f"no pair compares init seeds {relative_to} and {seed}")
    reference = {"pooled": {}, "suites": {}}
    for value in values.values():
        reference["pooled"].update(dict.fromkeys(value["pooled"], 0.))
        for suite, metrics in value["suites"].items():
            reference["suites"].setdefault(suite, {}).update(dict.fromkeys(metrics, 0.))
    values[str(relative_to)] = reference
    return {str(seed): values[str(seed)] for seed in seeds}


def recentred(values: Mapping[str, Mapping], seed: int) -> dict[str, dict]:
    """``values`` (``seed_values``) made relative to another of their seeds: each value minus ``seed``'s on the same
    metric (and suite); the values ``seed`` lacks are left out."""
    origin = values[str(seed)]
    return {name: {"pooled": {metric: value - origin["pooled"][metric] for metric, value in found["pooled"].items()
                              if metric in origin["pooled"]},
                   "suites": {suite: {metric: value - origin["suites"][suite][metric]
                                      for metric, value in metrics.items() if metric in origin["suites"].get(suite, {})}
                              for suite, metrics in found["suites"].items()}}
            for name, found in values.items()}


def seed_distribution(values: Mapping[str, Mapping], metric: str, suite: str | None = None) -> list[float]:
    """``metric``'s values over the seeds of ``values`` that have it: pooled, or ``suite``'s."""
    return [found["pooled"][metric] if suite is None else found["suites"][suite][metric]
            for found in values.values()
            if metric in (found["pooled"] if suite is None else found["suites"].get(suite, {}))]


def baseline_summary(values: Mapping[str, Mapping]) -> dict[str, dict]:
    """Each metric -> the ``mean`` and sample ``sd`` of its pooled values over the seeds of ``values`` that have it,
    and their number ``n`` (the sd is None for one seed)."""
    metrics = list(dict.fromkeys(metric for found in values.values() for metric in found["pooled"]))
    summary = {}
    for metric in metrics:
        found = seed_distribution(values, metric)
        summary[metric] = {"mean": statistics.fmean(found), "sd": statistics.stdev(found) if len(found) > 1 else None,
                           "n": len(found)}
    return summary


def decide(comparison: Mapping, band: Mapping[str, float] | None, metrics: Sequence[str]) -> dict[str, dict]:
    """Each of ``metrics`` -> its ``pooled`` Δ, its ``band`` (None without one), the suites counted for the sign rule
    (``suites``: those with enough clusters for a verdict, ``basis`` ``clusters``, or every suite that compares the
    metric when none has them, ``all``) and how many of them ``agree`` in sign with the pooled Δ, and the ``decision``
    (see the module docstring; None, and no basis, when no suite compares the metric)."""
    decisions = {}
    for metric in metrics:
        delta, limit = pooled(comparison, metric), (band or {}).get(metric)
        found = suite_deltas(comparison, metric)
        clustered = [suite for suite in found.values() if suite["clustered"]]
        counted, basis = (clustered, "clusters") if clustered else (list(found.values()), "all" if found else None)
        agree = sum(suite["delta"] * delta > 0 for suite in counted) if delta else 0
        if delta is None:
            decision = None
        elif limit is None:
            decision = UNCALIBRATED
        elif limit == 0:
            decision = ZERO_BAND
        else:
            decision = BEYOND if abs(delta) > limit and 3 * agree >= 2 * len(counted) else WITHIN
        decisions[metric] = {"pooled": delta, "band": limit, "agree": agree, "suites": len(counted), "basis": basis,
                             "decision": decision}
    return decisions
