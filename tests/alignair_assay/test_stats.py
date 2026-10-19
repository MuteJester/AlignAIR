import json
import math

from alignair_assay.stats import MetricSummary, by_slice, compare, holm, summarize, to_dict

NAN = float("nan")


def test_holm_known_example():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert out == {"a": True, "b": False, "c": False}


def test_bootstrap_deterministic():
    a = {f"c{i}": {"m": i % 3 * 0.1} for i in range(30)}
    b = {f"c{i}": {"m": i % 3 * 0.1 + (0.05 if i % 2 else -0.02)} for i in range(30)}
    r1 = compare(a, b, ["m"], lower_is_better=set())
    r2 = compare(a, b, ["m"], lower_is_better=set())
    assert r1["m"]["ci"] == r2["m"]["ci"]
    assert r1["m"]["p_boot"] == r2["m"]["p_boot"]
    json.dumps(to_dict(r1))


def test_coverage_reported():
    a = {f"c{i}": {"m": 1.0} for i in range(10)}
    b = {f"c{i}": {"m": 1.0 if i >= 3 else NAN} for i in range(10)}
    r = compare(a, b, ["m"], lower_is_better=set())["m"]
    assert r["coverage_b"] == 0.7
    assert r["coverage_a"] == 1.0
    assert r["n_paired"] == 7


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
    assert r["p_boot"] == 1.0
    assert r["verdict"] == "tie_or_inconclusive"
    assert r["ties"] == 20


def test_no_pairs_inconclusive():
    r = compare({"c1": {"m": 1.0}}, {"c2": {"m": 2.0}}, ["m"], lower_is_better=set())["m"]
    assert r["n_paired"] == 0 and r["ci"] is None and r["ci95"] is None
    assert r["p_boot"] == 1.0 and r["verdict"] == "tie_or_inconclusive"
