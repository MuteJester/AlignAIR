import json
import math
from statistics import NormalDist

import numpy as np

from alignair_assay.stats import MetricSummary, by_slice, compare, holm, summarize, to_dict

NAN = float("nan")


def test_holm_known_example():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert out == {"a": True, "b": False, "c": False}


def test_deterministic_and_json_ready():
    a = {f"c{i}": {"m": i % 3 * 0.1} for i in range(30)}
    b = {f"c{i}": {"m": i % 3 * 0.1 + (0.05 if i % 2 else -0.02)} for i in range(30)}
    r1 = compare(a, b, ["m"], lower_is_better=set())
    r2 = compare(a, b, ["m"], lower_is_better=set())
    assert r1 == r2
    json.dumps(to_dict(r1))


def test_coverage_and_paired_means_reported():
    a = {f"c{i}": {"m": float(i)} for i in range(10)}
    b = {f"c{i}": {"m": float(i) + 1.0 if i >= 3 else NAN} for i in range(10)}
    r = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert r["coverage_b"] == 0.7
    assert r["coverage_a"] == 1.0
    assert r["n_paired"] == 7
    # Each tool's own mean is over its own cases; the paired means are over the 7 shared cases.
    assert r["a"].mean == 4.5 and r["b"].mean == 7.0
    assert (r["mean_a_paired"], r["mean_b_paired"], r["delta"]) == (6.0, 7.0, 1.0)


def test_clear_winner():
    a = {f"c{i}": {"m": 0.2} for i in range(200)}
    b = {f"c{i}": {"m": 0.7} for i in range(200)}
    r = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert r["verdict"] == "b_better"
    assert r["wins"] == 200 and r["losses"] == 0
    assert math.isclose(r["delta"], 0.5)


def test_lower_is_better_direction():
    a = {f"c{i}": {"x_err": 5.0 + i % 3} for i in range(50)}
    b = {f"c{i}": {"x_err": 1.0 + i % 3} for i in range(50)}
    r = compare(a, b, ["x_err"], lower_is_better={"x_err"})["x_err"]
    assert r["verdict"] == "b_better"
    assert r["delta"] < 0 and r["wins"] == 50


def _binary_pairs(n_up, n_down, n_same):
    """a/b scores with n_up cases where only b is right, n_down where only a is, n_same both right."""
    pairs = [(0.0, 1.0)] * n_up + [(1.0, 0.0)] * n_down + [(1.0, 1.0)] * n_same
    a = {f"c{i}": {"m": x} for i, (x, _) in enumerate(pairs)}
    b = {f"c{i}": {"m": y} for i, (_, y) in enumerate(pairs)}
    return a, b


def test_binary_metric_uses_exact_mcnemar():
    a, b = _binary_pairs(15, 5, 30)
    r = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert r["test"] == "mcnemar_exact"
    # Two-sided exact binomial on 20 discordant pairs, 5 in the minority direction.
    assert math.isclose(r["p_value"], 2 * sum(math.comb(20, i) for i in range(6)) / 2 ** 20)
    a, b = _binary_pairs(0, 0, 30)
    assert compare(a, b, ["m"], lower_is_better=set())["m"]["p_value"] == 1.0


def test_binary_delta_ci_clipped_to_attainable_range():
    a = {f"c{i}": {"m": 1.0} for i in range(12)}
    b = {f"c{i}": {"m": 1.0 if i == 0 else 0.0} for i in range(12)}
    r = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert r["ci95"][0] == -1.0 and r["ci"][0] == -1.0 and r["ci95"][1] <= 1.0
    # Only the bound past -1 moves; the other keeps its normal-theory value (one metric: ci is the 95% interval).
    se = math.sqrt((1 / 12) / 12)
    assert math.isclose(r["ci95"][1], -11 / 12 + NormalDist().inv_cdf(0.975) * se)
    assert r["ci"] == r["ci95"]
    # Continuous metrics are not clipped.
    wide = compare({c: {"m": 0.0} for c in a}, {c: {"m": 3.0 * (i % 2)} for i, c in enumerate(a)}, ["m"],
                   lower_is_better=set())["m"]
    assert wide["ci95"][1] > 1.0


def test_continuous_metric_uses_paired_z():
    a = {f"c{i}": {"m": 0.0} for i in range(20)}
    b = {f"c{i}": {"m": 1.0 if i % 2 else 3.0} for i in range(20)}
    r = compare(a, b, ["m"], lower_is_better=set(), alpha=0.05)["m"]
    assert r["test"] == "paired_z"
    se = math.sqrt(20 / 19) / math.sqrt(20)
    assert math.isclose(r["p_value"], math.erfc(2.0 / se / math.sqrt(2)))
    z95 = NormalDist().inv_cdf(0.975)
    assert np.allclose(r["ci95"], (2.0 - z95 * se, 2.0 + z95 * se))
    # One metric in the family: the simultaneous interval is the 95% one.
    assert np.allclose(r["ci"], r["ci95"])
    wide = compare(a, b, ["m", "other"], lower_is_better=set(), alpha=0.05)["m"]
    z_family = NormalDist().inv_cdf(1 - 0.05 / 4)
    assert np.allclose(wide["ci"], (2.0 - z_family * se, 2.0 + z_family * se))


def test_fewer_than_ten_pairs_never_significant():
    a = {f"c{i}": {"m": 0.0} for i in range(9)}
    b = {f"c{i}": {"m": 1.0 + i} for i in range(9)}
    r = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert r["p_value"] == 1.0 and r["verdict"] == "tie_or_inconclusive"


def test_holm_changes_a_verdict_versus_uncorrected():
    # p = 0.041 for "m": significant alone at alpha 0.05, not after Holm with a second (null) metric.
    a, b = _binary_pairs(15, 5, 30)
    for c in a:
        a[c]["other"] = b[c]["other"] = 1.0
    alone = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert alone["p_value"] < 0.05 and alone["verdict"] == "b_better"
    family = compare(a, b, ["m", "other"], lower_is_better=set())
    assert family["m"]["p_value"] == alone["p_value"]
    assert family["m"]["verdict"] == "tie_or_inconclusive"


def _null_scores(rng, names, n, rates):
    """One tool's scores under the null: 0/1 accuracies at fixed rates, then normal errors, then error counts."""
    cols = [(rng.random(n) < r).astype(float) for r in rates]
    cols += [rng.normal(1.0, 1.0, n) for _ in range(15)]
    cols += [rng.poisson(0.8, n).astype(float) for _ in range(len(names) - len(cols))]
    rows = np.column_stack(cols).tolist()
    return {f"c{i}": dict(zip(names, row)) for i, row in enumerate(rows)}


def test_null_calibration():
    """Global null over a family shaped like the IGH comparison: family-wise error stays at alpha."""
    rng = np.random.default_rng(20261019)
    m, n, reps, alpha = 59, 400, 300, 0.05
    names = [f"m{k}" for k in range(m)]
    rates = rng.uniform(0.05, 0.95, size=30)
    family_errors = 0
    for _ in range(reps):
        a, b = _null_scores(rng, names, n, rates), _null_scores(rng, names, n, rates)
        res = compare(a, b, names, lower_is_better=set(), alpha=alpha)
        family_errors += any(e["verdict"] != "tie_or_inconclusive" for e in res.values())
    assert family_errors / reps <= 0.07, family_errors / reps


def test_absent_vs_nan_coverage():
    scores = {"c1": {"m": 1.0}, "c2": {"m": NAN}, "c3": {}, "c4": {"m": 3.0}}
    s = summarize(scores, ["c1", "c2", "c3", "c4", "missing"])["m"]
    assert isinstance(s, MetricSummary)
    assert s.n_applicable == 3 and s.n == 2
    assert s.mean == 2.0 and math.isclose(s.coverage, 2 / 3)


def test_summarize_applicable_override_and_empty_mean():
    scores = {"c1": {"m": NAN}, "c2": {}}
    s = summarize(scores, ["c1", "c2"], applicable={"m": {"c1", "c2"}})["m"]
    assert s.n_applicable == 2 and s.n == 0 and s.mean is None and s.coverage == 0.0


def test_by_slice_skips_unlabelled():
    scores = {"c1": {"m": 1.0}, "c2": {"m": 3.0}, "c3": {"m": 9.0}}
    out = by_slice(scores, {"c1": "x", "c2": "x", "c3": "y", "zz": "y"})
    assert out["x"]["m"].mean == 2.0 and out["y"]["m"].n == 1


def test_all_ties_p_one():
    a = {f"c{i}": {"m": 0.5} for i in range(20)}
    r = compare(a, dict(a), ["m"], lower_is_better=set())["m"]
    assert r["p_value"] == 1.0
    assert r["verdict"] == "tie_or_inconclusive"
    assert r["ties"] == 20


def test_no_pairs_inconclusive():
    r = compare({"c1": {"m": 1.0}}, {"c2": {"m": 2.0}}, ["m"], lower_is_better=set())["m"]
    assert r["n_paired"] == 0 and r["ci"] is None and r["ci95"] is None
    assert r["mean_a_paired"] is None and r["mean_b_paired"] is None and r["delta"] is None
    assert r["p_value"] == 1.0 and r["verdict"] == "tie_or_inconclusive"
