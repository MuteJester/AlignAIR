"""Secondary metrics (``secondary``), which the five headline metrics cannot see: read from the assay's cached
per-suite diagnostics of each model, never diagnosed again. ``d_short.abstain_ok`` is the first: a D presence gate can
only lower D top-1 on a short D, never raise it, so its benefit shows only as abstention there."""
import json

import pytest

from alignair_lab.baselines import null_band
from alignair_lab.noise import seed_values
from alignair_lab.secondary import METRIC, cached_comparison, cached_seed_values, merged, suite_scores

pytestmark = pytest.mark.component


def write_scores(root, key, scores):
    """A model's cached diagnostics under ``root`` (an assay run's ``diagnostics``): suite -> (mean, n) of METRIC."""
    for suite, (mean, n) in scores.items():
        directory = root / key / suite
        directory.mkdir(parents=True)
        found = {} if mean is None else {METRIC: {"mean": mean, "n": n, "n_applicable": n, "coverage": 1.0}}
        (directory / "suite.json").write_text(json.dumps({"variants": {"baseline": {"scores": found}}}))


def test_a_models_cached_scores_per_suite(tmp_path):
    write_scores(tmp_path, "cand", {"S1": (.8, 15), "S2": (.5, 4), "S3": (None, 0)})
    assert suite_scores(tmp_path, "cand", METRIC) == {"S1": {"mean": .8, "n": 15}, "S2": {"mean": .5, "n": 4}}
    assert suite_scores(tmp_path, "missing", METRIC) == {}


def test_a_comparison_of_cached_scores_is_shaped_as_the_assays(tmp_path):
    """The candidate's mean less the baseline's, per suite both score on the same reads; no test and no verdict, which
    the assay pairs no read-level verdict for."""
    write_scores(tmp_path, "cand", {"S1": (.8, 15), "S2": (.5, 4), "S3": (.25, 4)})
    write_scores(tmp_path, "base", {"S1": (.6, 15), "S2": (.75, 4), "S3": (.25, 8)})
    found = cached_comparison(tmp_path, "cand", "base", METRIC)
    assert set(found["suites"]) == {"S1", "S2"}                                     # S3: other reads
    assert found["suites"]["S1"]["headline"][METRIC] == {"delta": pytest.approx(.2), "n_paired": 15, "test": None,
                                                         "verdict": None}
    assert found["suites"]["S2"]["headline"][METRIC]["delta"] == pytest.approx(-.25)


def test_a_comparison_is_merged_into_another_without_changing_either():
    assay = {"baseline": "k", "suites": {"S1": {"cases": 9, "headline": {"clone_key.exact": {"delta": .1}}}}}
    extra = {"suites": {"S1": {"headline": {METRIC: {"delta": .2}}}, "S2": {"headline": {METRIC: {"delta": .3}}}}}
    found = merged(assay, extra)
    assert found["suites"]["S1"]["headline"] == {"clone_key.exact": {"delta": .1}, METRIC: {"delta": .2}}
    assert found["suites"]["S1"]["cases"] == 9 and "S2" not in found["suites"] and found["baseline"] == "k"
    assert assay["suites"]["S1"]["headline"] == {"clone_key.exact": {"delta": .1}}
    assert merged(assay, None) == assay


def test_a_calibrations_seeds_are_valued_from_their_cached_scores(tmp_path):
    """Each seed's Δs against the reference seed's screen, per suite and pooled over the suites by their reads."""
    write_scores(tmp_path, "k0", {"S1": (.6, 10), "S2": (.5, 4)})
    write_scores(tmp_path, "k1", {"S1": (.7, 10), "S2": (.25, 4)})
    write_scores(tmp_path, "k2", {"S1": (.5, 10), "S2": (.5, 4)})
    found = cached_seed_values(tmp_path, {"0": "k0", "1": "k1", "2": "k2"}, [0, 1, 2], relative_to=0, metric=METRIC)
    assert found["0"] == {"pooled": {METRIC: 0.}, "suites": {"S1": {METRIC: 0.}, "S2": {METRIC: 0.}}}
    assert found["1"]["suites"]["S1"][METRIC] == pytest.approx(.1)
    assert found["1"]["pooled"][METRIC] == pytest.approx((.1 * 10 - .25 * 4) / 14)
    assert cached_seed_values(tmp_path, {"0": "k0", "1": "k1"}, [0, 1, 2], relative_to=0, metric=METRIC) is None


def test_an_older_calibration_gets_the_metric_from_its_cached_scores_or_none(tmp_path):
    """Its pairs predate the metric: each seed's values are filled from the calibration's own cached diagnostics,
    when every seed's are there; otherwise the metric stays out, and screens weighed by it are uncalibrated on it."""
    metric = "clone_key.exact"
    pairs = [{"a": a, "b": b, "pooled": {metric: delta}, "suites": {"S1": {metric: delta}}}
             for a, b, delta in ((0, 1, .02), (0, 2, -.01), (1, 2, -.03))]
    protocol, builds = {"recipe": "v3", "cache": "c" * 64, "steps": 2, "schedule": None}, {"S1": "b" * 64}
    record = {"kind": "alignair_lab.null_band", "version": 1, "protocol": protocol, "options": {}, "suites": ["S1"],
              "suite_builds": builds, "max_cases": None, "init_seeds": [0, 1, 2], "pairs": pairs,
              "band": {metric: .03}, "code_sha256": "c" * 64,
              "screens": {str(seed): {"name": f"base{seed}", "model_key": f"k{seed}"} for seed in (0, 1, 2)}}
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "null_band.json").write_text(json.dumps(record))
    without = null_band(tmp_path, protocol, {}, ["S1"], None, builds)["seeds"]
    assert without == seed_values(pairs, [0, 1, 2], relative_to=0)
    diagnostics = calibration / "assay" / "diagnostics"
    for seed, mean in ((0, .6), (1, .7), (2, .4)):
        write_scores(diagnostics, f"k{seed}", {"S1": (mean, 10)})
    found = null_band(tmp_path, protocol, {}, ["S1"], None, builds)["seeds"]
    assert found["1"]["pooled"][METRIC] == pytest.approx(.1) and found["2"]["suites"]["S1"][METRIC] == \
        pytest.approx(-.2)
    assert found["1"]["pooled"][metric] == pytest.approx(.02)                        # the pairs' own, unchanged
