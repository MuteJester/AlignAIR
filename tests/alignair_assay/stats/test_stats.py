import hashlib
import json
import math
import random
from statistics import NormalDist

import numpy as np
import pytest

from alignair_assay.stats import MetricSummary, by_slice, compare, holm, read_changes, summarize, to_dict

pytestmark = pytest.mark.component

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


# --- cluster-aware comparisons ------------------------------------------------------------------------------------

# compare()'s output on _golden_data() before clusters existed (commit 4e3b920), as the SHA-256 of its JSON with
# sorted keys: with every cluster a singleton, the output must stay byte for byte what it was.
GOLDEN_SHA256 = "f88a108c58ab7b6689886d39a5be75846e3c251e9e50cceb84cc59a97d03a8d7"


def _golden_data():
    rng = random.Random(20261020)
    a, b = {}, {}
    for i in range(60):
        cid = f"c{i:03d}"
        a[cid] = {"bin": float(rng.random() < 0.6), "cont": rng.gauss(0.0, 1.0), "x_err": abs(rng.gauss(2, 1)),
                  "few": 1.0}
        b[cid] = {"bin": float(rng.random() < 0.75), "cont": rng.gauss(0.3, 1.0), "x_err": abs(rng.gauss(1.5, 1)),
                  "few": 0.0}
        if i % 7 == 0:
            b[cid]["cont"] = float("nan")
        if i >= 5:
            del a[cid]["few"]
    return a, b


def _without_clusters(result: dict) -> dict:
    return {name: {key: value for key, value in entry.items() if key != "n_clusters"}
            for name, entry in result.items()}


def test_singleton_clusters_keep_the_output_byte_for_byte():
    a, b = _golden_data()
    metrics = ["bin", "cont", "x_err", "few", "absent"]
    plain = compare(a, b, metrics, lower_is_better={"x_err"})
    text = json.dumps(to_dict(_without_clusters(plain)), sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == GOLDEN_SHA256
    # Every case its own cluster, by name, by omission or by an empty mapping: the same output.
    for clusters in ({}, {cid: f"k{cid}" for cid in a}, {"c000": "only"}):
        assert compare(a, b, metrics, lower_is_better={"x_err"}, clusters=clusters) == plain
    assert all(entry["n_clusters"] == entry["n_paired"] for entry in plain.values())


def _clustered(diffs_by_cluster: dict[str, list[int]]):
    """0/1 scores of tools a and b whose paired difference (b - a) per case is given, grouped by cluster: a
    difference of 1 is (0, 1), -1 is (1, 0) and 0 is (1, 1). Returns a, b and the case -> cluster mapping."""
    a, b, clusters = {}, {}, {}
    pair = {1: (0.0, 1.0), -1: (1.0, 0.0), 0: (1.0, 1.0)}
    for cluster, diffs in diffs_by_cluster.items():
        for k, diff in enumerate(diffs):
            cid = f"{cluster}:{k}"
            a[cid], b[cid] = {"m": pair[diff][0]}, {"m": pair[diff][1]}
            clusters[cid] = cluster
    return a, b, clusters


def test_one_clone_carrying_the_difference_is_not_significant():
    # The S9 shape: 20 reads of one clone turn correct, the other clones' reads do not change.
    five = {"k0": [1] * 20, **{f"k{k}": [0] * 20 for k in range(1, 5)}}
    a, b, clusters = _clustered(five)
    reads = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert reads["test"] == "mcnemar_exact" and reads["p_value"] == pytest.approx(2 / 2 ** 20)
    assert reads["verdict"] == "b_better"
    clustered = compare(a, b, ["m"], lower_is_better=set(), clusters=clusters)["m"]
    assert (clustered["test"], clustered["p_value"]) == ("too_few_clusters", 1.0)
    assert clustered["verdict"] == "tie_or_inconclusive"
    assert (clustered["n_paired"], clustered["n_clusters"], clustered["delta"]) == (100, 5, 0.2)
    # The same shape over 10 clones: enough clusters for a test, which one clone cannot carry (z = 20 / 20 = 1).
    ten = {"k0": [1] * 20, **{f"k{k}": [0] * 20 for k in range(1, 10)}}
    a, b, clusters = _clustered(ten)
    clustered = compare(a, b, ["m"], lower_is_better=set(), clusters=clusters)["m"]
    assert clustered["test"] == "clustered_mcnemar" and clustered["n_clusters"] == 10
    assert clustered["p_value"] == pytest.approx(math.erfc(1 / math.sqrt(2))) and clustered["p_value"] >= 0.3
    assert clustered["verdict"] == "tie_or_inconclusive"


def test_a_consistent_effect_across_clusters_is_significant():
    a, b, clusters = _clustered({f"k{k}": [1, 1, 0, 0] for k in range(30)})
    r = compare(a, b, ["m"], lower_is_better=set(), clusters=clusters)["m"]
    assert r["test"] == "clustered_mcnemar" and r["n_clusters"] == 30 and r["n_paired"] == 120
    assert r["p_value"] == pytest.approx(math.erfc(60 / math.sqrt(120) / math.sqrt(2)))
    assert r["p_value"] < 1e-6 and r["verdict"] == "b_better"


def test_clustered_z_and_cr1_interval_by_hand():
    # 6 clusters of 2 with D = 2, 3 clusters of 3 with D = 1, 3 single reads with D = -1: N = 24, K = 12,
    # sum D = 12, delta = 0.5, sum D^2 = 30; sum (D - n delta)^2 = 6 * 1 + 3 * 0.25 + 3 * 2.25 = 13.5.
    diffs = {**{f"two{k}": [1, 1] for k in range(6)}, **{f"three{k}": [1, 0, 0] for k in range(3)},
             **{f"one{k}": [-1] for k in range(3)}}
    a, b, clusters = _clustered(diffs)
    r = compare(a, b, ["m"], lower_is_better=set(), clusters=clusters)["m"]
    assert (r["n_paired"], r["n_clusters"], r["delta"]) == (24, 12, 0.5)
    assert r["test"] == "clustered_mcnemar"
    assert r["p_value"] == pytest.approx(math.erfc(12 / math.sqrt(30) / math.sqrt(2)))
    se = math.sqrt(12 / 11 * 13.5) / 24
    z95 = NormalDist().inv_cdf(0.975)
    assert np.allclose(r["ci95"], (0.5 - z95 * se, 0.5 + z95 * se)) and np.allclose(r["ci"], r["ci95"])
    # The paired means and the read-level counts do not depend on the clusters.
    assert (r["mean_a_paired"], r["mean_b_paired"]) == (9 / 24, 21 / 24)
    assert (r["wins"], r["losses"], r["ties"]) == (15, 3, 6)
    # A continuous metric is tested the same way, under its own name.
    cont_a = {cid: {"m": 0.0} for cid in a}
    cont_b = {cid: {"m": b[cid]["m"] - a[cid]["m"] + 0.5} for cid in a}
    cont = compare(cont_a, cont_b, ["m"], lower_is_better=set(), clusters=clusters)["m"]
    assert cont["test"] == "clustered_z" and cont["delta"] == pytest.approx(1.0)
    shifted = [d + 0.5 * n for d, n in ((2, 2),) * 6 + ((1, 3),) * 3 + ((-1, 1),) * 3]
    assert cont["p_value"] == pytest.approx(math.erfc(sum(shifted) / math.sqrt(sum(s * s for s in shifted))
                                                      / math.sqrt(2)))
    assert np.allclose(cont["ci95"], (1.0 - z95 * se, 1.0 + z95 * se))  # a shift moves the interval only


def test_the_interval_widens_on_positively_correlated_data():
    # Reads of a cluster move together: 8 of 20 clusters of 5 turn correct, the rest do not change.
    a, b, clusters = _clustered({f"k{k}": [int(k < 8)] * 5 for k in range(20)})
    reads = compare(a, b, ["m"], lower_is_better=set())["m"]
    clustered = compare(a, b, ["m"], lower_is_better=set(), clusters=clusters)["m"]
    assert reads["delta"] == clustered["delta"] == pytest.approx(0.4)
    width = {name: r["ci95"][1] - r["ci95"][0] for name, r in (("reads", reads), ("clusters", clustered))}
    assert width["clusters"] > 2 * width["reads"]
    assert clustered["p_value"] > reads["p_value"]


def test_too_few_pairs_comes_before_too_few_clusters():
    a, b, clusters = _clustered({"k0": [1] * 5, "k1": [1] * 4})
    r = compare(a, b, ["m"], lower_is_better=set(), clusters=clusters)["m"]
    assert (r["n_paired"], r["n_clusters"], r["test"], r["p_value"]) == (9, 2, "too_few_pairs", 1.0)


def test_read_changes_counts_the_reads_b_fixes_and_breaks():
    a = {"r1": {"hit": 0.0, "miss": 1.0, "dist": 2.0}, "r2": {"hit": 1.0, "miss": 0.0, "dist": 0.0},
         "r3": {"hit": 1.0, "miss": 0.0}, "r4": {"hit": 0.0, "miss": NAN}, "r5": {"hit": 1.0, "miss": 0.0}}
    b = {"r1": {"hit": 1.0, "miss": 0.0, "dist": 1.0}, "r2": {"hit": 0.0, "miss": 1.0, "dist": 0.0},
         "r3": {"hit": 1.0, "miss": 0.0}, "r4": {"hit": 0.0, "miss": 1.0}, "r6": {"hit": 1.0, "miss": 0.0}}
    found = read_changes(a, b, ["hit", "miss", "dist"], lower_is_better={"miss"}, examples=1)
    # a wrong and b right is fixed (oriented: for a lower-is-better metric, 1 -> 0); the other way is broken
    assert found["metrics"]["hit"] == {"n_paired": 4, "fixed": 1, "broken": 1, "fixed_examples": ["r1"],
                                       "broken_examples": ["r2"]}
    assert found["metrics"]["miss"] == {"n_paired": 3, "fixed": 1, "broken": 1, "fixed_examples": ["r1"],
                                        "broken_examples": ["r2"]}
    # a metric that is not 0/1 on its paired reads has no fixed or broken read
    assert "dist" not in found["metrics"]
    assert found["reads"] == {"fixed": 1, "broken": 1, "fixed_examples": ["r1"], "broken_examples": ["r2"]}
    same = read_changes(a, a, ["hit", "miss", "dist"], lower_is_better={"miss"})
    assert same["reads"] == {"fixed": 0, "broken": 0, "fixed_examples": [], "broken_examples": []}
    assert {name: (m["fixed"], m["broken"]) for name, m in same["metrics"].items()} == {"hit": (0, 0),
                                                                                         "miss": (0, 0)}
    assert (same["a_only"], same["b_only"]) == (0, 0)
    json.dumps(found)


def test_read_changes_examples_are_evenly_spaced_and_counted_per_group():
    # 25 reads fixed (c00..c24) and 5 broken (c30..c34): the examples are spread over the sorted ids, not the first
    # ones (a suite lists its cases stratum by stratum), and with groups each group's reads fixed and broken count
    a = {f"c{i:02d}": {"hit": 0.0 if i < 25 else 1.0} for i in range(35)}
    b = {f"c{i:02d}": {"hit": 1.0 if i < 30 else 0.0} for i in range(35)}
    groups = {case: "s1" if int(case[1:]) < 20 else "s2" for case in a}
    found = read_changes(a, b, ["hit"], lower_is_better=set(), examples=5, groups=groups)
    fixed = [f"c{i:02d}" for i in range(25)]
    assert found["metrics"]["hit"]["fixed_examples"] == [fixed[i * 25 // 5] for i in range(5)]
    assert found["metrics"]["hit"]["fixed_examples"] == ["c00", "c05", "c10", "c15", "c20"]
    assert found["reads"]["fixed_examples"] == ["c00", "c05", "c10", "c15", "c20"]
    assert found["reads"]["broken_examples"] == ["c30", "c31", "c32", "c33", "c34"]  # five of five
    assert found["reads"]["groups"] == {"s1": {"fixed": 20, "broken": 0}, "s2": {"fixed": 5, "broken": 5}}
    # without groups the reads carry no per-group counts
    assert "groups" not in read_changes(a, b, ["hit"], lower_is_better=set())["reads"]
    json.dumps(found)


def test_read_changes_counts_the_cases_of_one_side_only():
    # r5 is only in a and r6, r7 only in b: they are not paired, and not silently dropped either (a NaN is a paired
    # case's missing value, not a missing case)
    a = {"r1": {"hit": 0.0}, "r4": {"hit": NAN}, "r5": {"hit": 1.0}}
    b = {"r1": {"hit": 1.0}, "r4": {"hit": 1.0}, "r6": {"hit": 1.0}, "r7": {}}
    found = read_changes(a, b, ["hit"], lower_is_better=set(), examples=1)
    assert (found["a_only"], found["b_only"]) == (1, 2)
    assert (found["a_only_examples"], found["b_only_examples"]) == (["r5"], ["r6"])
    assert found["metrics"]["hit"]["n_paired"] == 1
    json.dumps(found)
