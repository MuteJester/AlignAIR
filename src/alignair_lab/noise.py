"""Run-to-run training noise, and the rule that decides whether a screen's Δ is beyond it. Pure: the assay's comparison
summaries in (``evaluate(..., against=...)``'s ``result.against["summary"]``), numbers out.

Two trainings of one architecture on the same records differ: GPU kernels are not deterministic, and replicate runs
draw other initial weights. A candidate's Δ against a baseline screen holds that noise beside the candidate's own
effect, and the assay's CIs, which resample the reads of two fixed checkpoints, leave it out. ``calibrate`` measures
it with baseline screens of several init seeds, compared pair by pair. Per metric:

- ``pooled``: a comparison's Δ (candidate − baseline) pooled over its suites, each weighted by its paired reads.
- ``null_band``: the largest |pooled Δ| of the pairs of baseline screens, kept as a printed reference.
- ``seed_values``: each baseline seed's pooled (and per-suite) value against the registered baseline screen, from its
  pair with that screen's seed: the baseline's distribution, of which the registered screen is one draw.
- ``decide``: the candidate's seeds against the baseline's. Each candidate seed's pooled Δ is against the baseline
  screen, as each baseline seed's value is, so the two distributions compare on one scale: the diff (candidate mean −
  baseline mean) over Welch's standard error, sqrt(sc²/k + sb²/n), is z (``welch``; with one candidate seed its
  spread is taken to be the baseline's, sc = sb). The candidate is ``beyond noise`` when |z| ≥ ``Z_LIMIT`` and at
  least 2/3 of the suites agree with the diff's sign: each suite's mean Δ over the candidate's seeds less the
  baseline seeds' mean there, counting the suites with enough clusters for the assay's verdict on the metric (its test
  is not ``too_few_clusters``: fewer than ``stats.compare.MIN_CLUSTERS``) in every seed's comparison, or every suite
  that compares it when none has them (the ``basis``: ``clusters`` or ``all``); otherwise ``within noise``;
  ``uncalibrated`` with fewer than ``MIN_SEEDS`` baseline seeds of the metric, and ``uncalibrated: spread 0`` when the
  standard error is 0: baseline seeds that never differed on it bound nothing.

Comparing with the registered baseline screen alone would weigh the candidate against one draw of the baseline's
distribution: a calibration of six seeds found the registered one the best or near-best on every headline metric,
so every candidate compared with it alone was biased down.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

BEYOND, WITHIN, UNCALIBRATED = "beyond noise", "within noise", "uncalibrated"
ZERO_SPREAD = "uncalibrated: spread 0"
MIN_SEEDS = 3                             # baseline seeds a calibration needs for a decision
Z_LIMIT = 2.0                             # |z| at or above which a diff is beyond noise (with the sign rule)
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


def welch(candidate: Sequence[float], baseline: Sequence[float]) -> dict:
    """The candidate's values against the baseline's: their means and sample SDs, their numbers ``k`` and ``n``, the
    ``diff`` (candidate mean − baseline mean), Welch's standard error ``se`` = sqrt(sc²/k + sb²/n) and ``z`` = diff /
    se (None when se is 0). With one candidate value its spread is unmeasured (``candidate_sd`` None) and is taken to
    be the baseline's: sc = sb. ValueError without a candidate value or with fewer than two baseline values."""
    k, n = len(candidate), len(baseline)
    if k < 1:
        raise ValueError("Welch's comparison needs a candidate value")
    if n < 2:
        raise ValueError(f"Welch's comparison needs two baseline values or more, got {n}")
    sb = statistics.stdev(baseline)
    measured = statistics.stdev(candidate) if k > 1 else None
    sc = sb if measured is None else measured
    se = math.sqrt(sc ** 2 / k + sb ** 2 / n)
    diff = statistics.fmean(candidate) - statistics.fmean(baseline)
    return {"candidate_mean": statistics.fmean(candidate), "candidate_sd": measured,
            "baseline_mean": statistics.fmean(baseline), "baseline_sd": sb, "k": k, "n": n, "diff": diff, "se": se,
            "z": diff / se if se else None}


def _suite_signs(comparisons: Sequence[Mapping], baseline: Mapping[str, Mapping] | None, metric: str,
                 direction: float | None) -> tuple[int, int, str | None]:
    """``(agree, counted, basis)`` of the sign rule: each suite's mean Δ over the comparisons that compare it, less the
    baseline seeds' mean in that suite (0 without them), against ``direction``'s sign."""
    found: dict[str, list[dict]] = {}
    for comparison in comparisons:
        for name, suite in suite_deltas(comparison, metric).items():
            found.setdefault(name, []).append(suite)
    clustered = {name for name, suites in found.items() if all(suite["clustered"] for suite in suites)}
    counted, basis = (clustered, "clusters") if clustered else (set(found), "all" if found else None)
    agree = 0
    for name in counted:
        shift = seed_distribution(baseline, metric, name) if baseline else []
        delta = statistics.fmean(suite["delta"] for suite in found[name]) - (statistics.fmean(shift) if shift else 0.)
        agree += bool(direction) and delta * direction > 0
    return agree, len(counted), basis


def decide(comparisons: Sequence[Mapping], baseline: Mapping[str, Mapping] | None, metrics: Sequence[str], *,
           band: Mapping[str, float] | None = None) -> dict[str, dict]:
    """Each of ``metrics`` -> the candidate's pooled Δ of each of its init seeds' ``comparisons`` (``candidate``;
    each against the baseline screen) and their mean (``pooled``), ``welch``'s numbers against the ``baseline``
    seeds' pooled values (each seed's Δ against the same baseline screen, ``seed_values``; None without a
    calibration), the reference ``band`` (None without one), the suites counted for the sign rule (``suites``, of
    ``basis`` ``clusters`` or ``all``) and how many ``agree`` with the diff's sign, and the ``decision`` (see the module
    docstring; None, and no basis, when no suite compares the metric)."""
    decisions = {}
    for metric in metrics:
        candidate = [value for value in (pooled(comparison, metric) for comparison in comparisons) if value is not None]
        values = seed_distribution(baseline, metric) if baseline else []
        found = {"candidate": candidate, "pooled": statistics.fmean(candidate) if candidate else None,
                 "candidate_sd": None, "baseline_mean": None, "baseline_sd": None, "k": len(candidate),
                 "n": len(values), "diff": None, "se": None, "z": None, "band": (band or {}).get(metric)}
        calibrated = bool(candidate) and len(values) >= MIN_SEEDS
        if calibrated:
            found.update(welch(candidate, values))
            del found["candidate_mean"]
        direction = found["diff"] if calibrated else found["pooled"]
        found["agree"], found["suites"], found["basis"] = _suite_signs(comparisons, baseline if calibrated else None,
                                                                       metric, direction)
        if not candidate:
            decision = None
        elif not calibrated:
            decision = UNCALIBRATED
        elif found["z"] is None:
            decision = ZERO_SPREAD
        else:
            decision = (BEYOND if abs(found["z"]) >= Z_LIMIT and 3 * found["agree"] >= 2 * found["suites"]
                        else WITHIN)
        decisions[metric] = {**found, "decision": decision}
    return decisions
