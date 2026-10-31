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
  seeds' Δs there against the baseline seeds' values there, with the one-sided p of a fall from Student's t at the
  Welch–Satterthwaite df (n − 1 for one candidate seed). A suite regresses when Holm's step-down, at family α
  ``FAMILY_ALPHA`` over the metric's suites that have a p, finds its fall significant and the fall is
  ``MIN_REGRESSION`` (1 pp) or more. A limited suite, with too few pairs or clusters for the assay's verdict on the
  metric (``too_few_pairs``, ``too_few_clusters``) in some seed's comparison, is reported and flagged but stays out of
  the family and never regresses. A suite whose z is undefined (its baseline seeds have no spread, or it has one
  baseline value) is out of the family: its fall of ``MIN_REGRESSION`` or more regresses when the assay's own paired
  verdict on the suite's reads is worse (``WORSE_READS``) in a majority of the candidate seeds' comparisons, and is
  listed as unmeasured (``unmeasured``) otherwise. The p is t's, not the normal's: from 1–3 candidate seeds and 6
  baseline seeds,
  Welch's z has heavy tails, and the normal's p would let Holm call some suite of eleven a regression in about a
  quarter of the screens without one. The pooled z is read from Student's t at its df as well (``pooled_test``),
  two-sided at ``POOLED_ALPHA``: z ≥ 2 alone would call about 1 in 20 candidates without an effect better. The
  decision: ``beyond noise (better)`` when the one-sided p of a gain is .025 or less and no suite regresses, ``mixed``
  when it is and some suite does (``regressions`` names them), ``beyond noise (worse)`` when the one-sided p of a
  loss is .025 or less, and ``within noise`` otherwise; the ``regressions`` are reported whatever the decision;
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

from scipy import stats

BETTER, WORSE = "beyond noise (better)", "beyond noise (worse)"
WITHIN, MIXED, UNCALIBRATED = "within noise", "mixed", "uncalibrated"
ZERO_SPREAD = "uncalibrated: spread 0"
MIN_SEEDS = 3                             # baseline seeds a calibration needs for a decision
POOLED_ALPHA = 0.05                       # the pooled verdict's two-sided level: .025 in each tail of Student's t
FAMILY_ALPHA = 0.05                       # Holm's family-wise level over a metric's suites, for their falls
MIN_REGRESSION = 0.01                     # the smallest fall of a suite that counts as a regression: 1 pp
# The assay's tests when a metric's pairs in a suite are too few, or fall in too few clusters, for a verdict.
FEW_CLUSTERS, FEW_PAIRS = "too_few_clusters", "too_few_pairs"
WORSE_READS = "a_better"                  # the assay's paired verdict on a suite's reads: the baseline (a) is better


def suite_deltas(comparison: Mapping, metric: str) -> dict[str, dict]:
    """Each suite that compares ``metric`` on paired reads -> its ``delta``, ``n_paired`` and ``clustered``: whether
    its pairs are enough, in enough clusters, for the assay's verdict (its test is neither ``too_few_clusters`` nor
    ``too_few_pairs``)."""
    found = {}
    for name, suite in comparison["suites"].items():
        entry = (suite.get("headline") or {}).get(metric)
        if entry and entry.get("delta") is not None and entry.get("n_paired"):
            found[name] = {"delta": entry["delta"], "n_paired": entry["n_paired"],
                           "clustered": entry.get("test") not in (FEW_CLUSTERS, FEW_PAIRS),
                           "verdict": entry.get("verdict")}
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
    ``diff`` (candidate mean − baseline mean), Welch's standard error ``se`` = sqrt(sc²/k + sb²/n), ``z`` = diff /
    se and its degrees of freedom ``df`` (Welch–Satterthwaite; both None when se is 0). With one candidate value its
    spread is unmeasured (``candidate_sd`` None) and is taken to be the baseline's, sc = sb, so df is n − 1.
    ValueError without a candidate value or with fewer than two baseline values."""
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
    if not se:
        df = None
    elif measured is None:
        df = n - 1                        # the only spread estimate is the baseline's
    else:
        a, b = sc ** 2 / k, sb ** 2 / n
        df = (a + b) ** 2 / (a ** 2 / (k - 1) + b ** 2 / (n - 1))
    return {"candidate_mean": statistics.fmean(candidate), "candidate_sd": measured,
            "baseline_mean": statistics.fmean(baseline), "baseline_sd": sb, "k": k, "n": n, "diff": diff, "se": se,
            "z": diff / se if se else None, "df": df}


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
                 "n": len(values), "diff": None, "se": None, "z": None, "df": None,
                 "band": (band or {}).get(metric)}
        calibrated = bool(candidate) and len(values) >= MIN_SEEDS
        found.update(p=None, p_better=None, p_worse=None)
        if calibrated:
            found.update(pooled_test(candidate, values))
            del found["candidate_mean"]
        direction = found["diff"] if calibrated else found["pooled"]
        found["agree"], found["suites"], found["basis"] = _suite_signs(comparisons, baseline if calibrated else None,
                                                                       metric, direction)
        found["suite_z"] = suite_tests(comparisons, baseline, metric) if calibrated else {}
        found["regressions"] = [name for name, test in found["suite_z"].items() if test["regresses"]]
        found["unmeasured"] = [name for name, test in found["suite_z"].items() if test["unmeasured_fall"]]
        if not candidate:
            decision = None
        elif not calibrated:
            decision = UNCALIBRATED
        elif found["z"] is None:
            decision = ZERO_SPREAD
        elif found["p_better"] <= POOLED_ALPHA / 2:
            decision = MIXED if found["regressions"] else BETTER
        elif found["p_worse"] <= POOLED_ALPHA / 2:
            decision = WORSE
        else:
            decision = WITHIN
        decisions[metric] = {**found, "decision": decision}
    return decisions


def pooled_test(candidate: Sequence[float], baseline: Sequence[float]) -> dict:
    """``welch`` of the candidate's values against the baseline's, with the one-sided p of a gain (``p_better``) and of
    a loss (``p_worse``) from Student's t at its df, and their two-sided ``p``; the three None when z is."""
    found = welch(candidate, baseline)
    if found["z"] is None:
        return {**found, "p": None, "p_better": None, "p_worse": None}
    better, worse = float(stats.t.sf(found["z"], found["df"])), float(stats.t.cdf(found["z"], found["df"]))
    return {**found, "p": min(1., 2 * min(better, worse)), "p_better": better, "p_worse": worse}


def suite_tests(comparisons: Sequence[Mapping], baseline: Mapping[str, Mapping], metric: str) -> dict[str, dict]:
    """Each suite the candidate's ``comparisons`` compare ``metric`` in -> its own ``welch`` test: the candidate seeds'
    Δs there against the ``baseline`` seeds' values there (``diff``, ``z``, ``df``, ``k``, ``n``; with one candidate
    seed its spread is taken to be the baseline's), z None with fewer than two baseline values or no spread, and the
    one-sided ``p`` of a fall (Student's t at df); whether it is ``limited`` (too few pairs or clusters for the assay's
    verdict in some seed's comparison); whether it is ``significant`` by Holm's step-down at family α
    ``FAMILY_ALPHA`` over the suites with a p that are not limited; and whether it ``regresses``: significant and down
    by ``MIN_REGRESSION`` or more."""
    found: dict[str, list[dict]] = {}
    for comparison in comparisons:
        for name, suite in suite_deltas(comparison, metric).items():
            found.setdefault(name, []).append(suite)
    tests = {}
    for name, suites in found.items():
        values = seed_distribution(baseline, metric, name)
        candidate = [suite["delta"] for suite in suites]
        test = {"diff": None, "z": None, "df": None, "p": None, "k": len(candidate), "n": len(values)}
        if values:
            test["diff"] = statistics.fmean(candidate) - statistics.fmean(values)
        if len(values) >= 2:
            numbers = welch(candidate, values)
            test.update(diff=numbers["diff"], z=numbers["z"], df=numbers["df"])
            if numbers["z"] is not None:
                test["p"] = float(stats.t.cdf(numbers["z"], numbers["df"]))
        limited = not all(suite["clustered"] for suite in suites)
        unmeasured = test["z"] is None and test["diff"] is not None
        falls = unmeasured and not limited and _falls(test["diff"])
        worse_reads = 2 * sum(suite["verdict"] == WORSE_READS for suite in suites) > len(suites)
        tests[name] = {**test, "limited": limited, "spread_unmeasured": unmeasured, "significant": False,
                       "regresses": falls and worse_reads, "unmeasured_fall": falls and not worse_reads}
    family = sorted((test["p"], name) for name, test in tests.items() if test["p"] is not None and not test["limited"])
    for step, (p, name) in enumerate(family):
        if p > FAMILY_ALPHA / (len(family) - step):
            break
        tests[name]["significant"] = True
        tests[name]["regresses"] = _falls(tests[name]["diff"])
    return tests


def _falls(diff: float) -> bool:
    """Whether a suite's diff is a fall of ``MIN_REGRESSION`` or more."""
    return diff <= -MIN_REGRESSION + 1e-12
