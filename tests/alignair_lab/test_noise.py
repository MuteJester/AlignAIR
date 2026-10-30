"""Run-to-run training noise: a comparison's Δ pooled over its suites by paired reads, the null band of a calibration's
pairwise comparisons, and the decision rule a candidate's headline applies, on fixture comparison summaries."""
import pytest

from alignair_lab.noise import decide, null_band, pair, pooled

pytestmark = pytest.mark.component

METRIC = "junction.nt_exact"


def entry(delta, n_paired=100, ci=True, test="clustered_mcnemar"):
    return {"delta": delta, "n_paired": n_paired, "ci95": [delta - .01, delta + .01] if ci else None,
            "verdict": "tie_or_inconclusive", "test": test}


FEW = "too_few_clusters"                    # the assay's verdict threshold: fewer than MIN_CLUSTERS clusters


def comparison(*suites, metric=METRIC):
    """A comparison summary of one suite per entry (None: the suite does not compare ``metric``)."""
    return {"suites": {f"S{n}": {"headline": {} if found is None else {metric: found}}
                       for n, found in enumerate(suites, 1)}}


def test_the_pooled_delta_weights_each_suite_by_its_paired_reads():
    found = comparison(entry(.10, 100), entry(-.02, 300), None, entry(.5, 0), {"delta": None, "n_paired": 0})
    assert pooled(found, METRIC) == pytest.approx((.10 * 100 - .02 * 300) / 400)
    assert pooled(comparison(None, None), METRIC) is None


def test_the_null_band_is_the_largest_pooled_delta_between_two_baseline_screens_and_keeps_each_suites():
    pairs = [pair(comparison(entry(.01), entry(.03))), pair(comparison(entry(-.05), entry(-.01))),
             pair(comparison(entry(.02), None))]
    assert pairs[0] == {"pooled": {METRIC: pytest.approx(.02)}, "suites": {"S1": {METRIC: .01}, "S2": {METRIC: .03}}}
    assert pairs[2]["suites"] == {"S1": {METRIC: .02}}
    assert null_band(pairs) == {METRIC: pytest.approx(.03)}


def test_a_metric_no_suite_of_a_pair_compares_has_no_band_from_that_pair():
    pairs = [pair(comparison(None)), pair(comparison(entry(-.04)))]
    assert pairs[0] == {"pooled": {}, "suites": {}}
    assert null_band(pairs) == {METRIC: pytest.approx(.04)}


def decision(*suites, band=.02):
    return decide(comparison(*suites), None if band is None else {METRIC: band}, [METRIC])[METRIC]


def test_beyond_noise_needs_a_pooled_delta_above_the_band_and_two_thirds_of_the_suites_signs():
    found = decision(entry(.05), entry(.04), entry(-.01))
    assert found == {"pooled": pytest.approx(.08 / 3), "band": .02, "agree": 2, "suites": 3, "basis": "clusters",
                     "decision": "beyond noise"}
    assert decision(entry(.05), entry(.04), entry(-.01), band=.03)["decision"] == "within noise"
    assert decision(entry(.08), entry(-.01))["decision"] == "within noise"            # half the signs
    assert decision(entry(-.06), entry(-.03), entry(.01))["decision"] == "beyond noise"   # either direction


def test_a_pooled_delta_equal_to_the_band_is_within_noise():
    assert decision(entry(.02), entry(.02))["decision"] == "within noise"


def test_the_signs_count_the_suites_with_enough_clusters_or_every_suite_when_none_has_them():
    """Enough clusters is the assay's own threshold for a verdict (a CI needs only two clusters)."""
    clustered = decision(entry(.06), entry(.06), entry(-.01, test=FEW), entry(-.01, test=FEW))
    assert (clustered["agree"], clustered["suites"], clustered["basis"], clustered["decision"]) == \
        (2, 2, "clusters", "beyond noise")
    assert decision(entry(.06, ci=False), entry(-.01), entry(-.01))["suites"] == 3        # a CI is not the rule
    fallback = decision(entry(.06, test=FEW), entry(-.01, test=FEW), entry(-.01, test=FEW))
    assert (fallback["agree"], fallback["suites"], fallback["basis"], fallback["decision"]) == \
        (1, 3, "all", "within noise")


def test_a_suite_with_no_change_does_not_agree():
    found = decision(entry(.09), entry(0.), entry(0.))
    assert (found["agree"], found["suites"], found["decision"]) == (1, 3, "within noise")


def test_without_a_band_nothing_is_marked():
    found = decision(entry(.5), entry(.5), band=None)
    assert (found["band"], found["decision"]) == (None, "uncalibrated")
    assert decide(comparison(entry(.5)), {"other.metric": .01}, [METRIC])[METRIC]["decision"] == "uncalibrated"


def test_a_band_of_zero_marks_nothing():
    """Baseline screens that never differed on a metric bound nothing: any Δ of agreeing signs would exceed 0."""
    found = decision(entry(.5), entry(.5), band=0.)
    assert (found["band"], found["decision"]) == (0., "uncalibrated: band 0")


def test_a_metric_no_suite_compares_has_no_decision():
    assert decision(None, None) == {"pooled": None, "band": .02, "agree": 0, "suites": 0, "basis": None,
                                    "decision": None}
