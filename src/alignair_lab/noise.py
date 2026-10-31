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
  spread is taken to be the baseline's, sc = sb). Each suite is tested the same way (``suite_tests``): the candidate
  seeds' Δs there against the baseline seeds' values there. A suite regresses at z ≤ −``Z_LIMIT``, unless it is
  limited: too few pairs or clusters for the assay's verdict on the metric (``too_few_pairs``, ``too_few_clusters``)
  in some seed's comparison, which is reported, flagged, and never counted. The decision:
  ``beyond noise (better)`` at z ≥ ``Z_LIMIT`` with no suite regressing, ``mixed`` at z ≥ ``Z_LIMIT`` with some
  (``regressions`` names them), ``beyond noise (worse)`` at z ≤ −``Z_LIMIT``, and ``within noise`` otherwise;
  ``uncalibrated`` with fewer than ``MIN_SEEDS`` baseline seeds of the metric, and ``uncalibrated: spread 0`` when the
  standard error is 0: baseline seeds that never differed on it bound nothing. A gain on the hard suites while the
  easy ones sit saturated at Δ 0 is a gain; what the suites guard against is one suite traded for another.
  The suites whose diff has the pooled diff's sign (``agree``, of the suites with enough pairs and clusters in every
  seed's comparison, or of every suite when none has them: the ``basis``) are kept as information.

Comparing with the registered baseline screen alone would weigh the candidate against one draw of the baseline's
distribution: a calibration of six seeds found the registered one the best or near-best on every headline metric,
so every candidate compared with it alone was biased down.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

BETTER, WORSE = "beyond noise (better)", "beyond noise (worse)"
WITHIN, MIXED, UNCALIBRATED = "within noise", "mixed", "uncalibrated"
ZERO_SPREAD = "uncalibrated: spread 0"
MIN_SEEDS = 3                             # baseline seeds a calibration needs for a decision
Z_LIMIT = 2.0                             # |z| from which a diff is beyond noise; a suite at z <= -Z_LIMIT regresses
# The assay's tests when a metric's pairs in a suite are too few, or fall in too few clusters, for a verdict.
FEW_CLUSTERS, FEW_PAIRS = "too_few_clusters", "too_few_pairs"


def suite_deltas(comparison: Mapping, metric: str) -> dict[str, dict]:
    """Each suite that compares ``metric`` on paired reads -> its ``delta``, ``n_paired`` and ``clustered``: whether
    its pairs are enough, in enough clusters, for the assay's verdict (its test is neither ``too_few_clusters`` nor
    ``too_few_pairs``)."""
    found = {}
    for name, suite in comparison["suites"].items():
        entry = (suite.get("headline") or {}).get(metric)
        if entry and entry.get("delta") is not None and entry.get("n_paired"):
            found[name] = {"delta": entry["delta"], "n_paired": entry["n_paired"],
                           "clustered": entry.get("test") not in (FEW_CLUSTERS, FEW_PAIRS)}
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
    calibration), the reference ``band`` (None without one), each suite's own Welch test (``suite_z``,
    ``suite_tests``; empty without a calibration) and the suites that regress (``regressions``), the suites counted for
    the signs (``suites``, of ``basis`` ``clusters`` or ``all``) and how many ``agree`` with the diff's sign, kept as
    information, and the ``decision`` (see the module docstring; None, and no basis, when no suite compares the
    metric)."""
    decisions = {}
    for metric in metrics:
        candidate = [value for value in (pooled(comparison, metric) for comparison in comparisons) if value is not None]
        values = seed_distribution(baseline, metric) if baseline else []
        found = {"candidate": candidate, "pooled": statistics.fmean(candidate) if candidate else None,
                 "candidate_sd": statistics.stdev(candidate) if len(candidate) > 1 else None,
                 "baseline_mean": None, "baseline_sd": None, "k": len(candidate),
                 "n": len(values), "diff": None, "se": None, "z": None, "band": (band or {}).get(metric)}
        calibrated = bool(candidate) and len(values) >= MIN_SEEDS
        if calibrated:
            found.update(welch(candidate, values))
            del found["candidate_mean"]
        direction = found["diff"] if calibrated else found["pooled"]
        found["agree"], found["suites"], found["basis"] = _suite_signs(comparisons, baseline if calibrated else None,
                                                                       metric, direction)
        found["suite_z"] = suite_tests(comparisons, baseline, metric) if calibrated else {}
        found["regressions"] = [name for name, test in found["suite_z"].items()
                                if not test["limited"] and test["z"] is not None and test["z"] <= -Z_LIMIT]
        if not candidate:
            decision = None
        elif not calibrated:
            decision = UNCALIBRATED
        elif found["z"] is None:
            decision = ZERO_SPREAD
        elif found["z"] >= Z_LIMIT:
            decision = MIXED if found["regressions"] else BETTER
        elif found["z"] <= -Z_LIMIT:
            decision = WORSE
        else:
            decision = WITHIN
        decisions[metric] = {**found, "decision": decision}
    return decisions


def suite_tests(comparisons: Sequence[Mapping], baseline: Mapping[str, Mapping], metric: str) -> dict[str, dict]:
    """Each suite the candidate's ``comparisons`` compare ``metric`` in -> its own ``welch`` test: the candidate seeds'
    Δs there against the ``baseline`` seeds' values there (``diff``, ``z``, ``k``, ``n``; with one candidate seed its
    spread is taken to be the baseline's), z None with fewer than two baseline values or no spread; and whether it is
    ``limited``: too few pairs or clusters for the assay's verdict in some seed's comparison, so it never counts as a
    regression."""
    found: dict[str, list[dict]] = {}
    for comparison in comparisons:
        for name, suite in suite_deltas(comparison, metric).items():
            found.setdefault(name, []).append(suite)
    tests = {}
    for name, suites in found.items():
        values = seed_distribution(baseline, metric, name)
        candidate = [suite["delta"] for suite in suites]
        test = {"diff": None, "z": None, "k": len(candidate), "n": len(values)}
        if len(values) >= 2:
            numbers = welch(candidate, values)
            test.update(diff=numbers["diff"], z=numbers["z"])
        tests[name] = {**test, "limited": not all(suite["clustered"] for suite in suites)}
    return tests
