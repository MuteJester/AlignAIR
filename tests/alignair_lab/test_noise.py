"""Run-to-run training noise: a comparison's Δ pooled over its suites by paired reads, the null band of a calibration's
pairwise comparisons, each baseline seed's values, and the decision rule a candidate's headline applies (its seeds
against the baseline's, by Welch's z, and the suites' signs), on fixture comparison summaries."""
import math
import statistics

import pytest

from alignair_lab.noise import baseline_summary, decide, null_band, pair, pooled, recentred, seed_values, welch

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


def calibration_pairs():
    """The three pairs of a calibration of init seeds 0, 1 and 2, each the later seed's Δs against the earlier's."""
    return [{"a": 0, "b": 1, **pair(comparison(entry(.01), entry(.03)))},
            {"a": 0, "b": 2, **pair(comparison(entry(-.05), entry(-.01)))},
            {"a": 1, "b": 2, **pair(comparison(entry(-.06), entry(-.04)))}]


def test_each_baseline_seed_is_valued_relative_to_a_reference_seed_from_its_pairs():
    """Seed s's value is the pooled (and each suite's) Δ of the pair (reference, s), negated for the pair (s,
    reference); the reference's own is 0 on every metric and suite the others have."""
    pairs = calibration_pairs()
    values = seed_values(pairs, [0, 1, 2], relative_to=0)
    assert list(values) == ["0", "1", "2"]
    assert values["0"] == {"pooled": {METRIC: 0.}, "suites": {"S1": {METRIC: 0.}, "S2": {METRIC: 0.}}}
    assert values["1"] == {"pooled": {METRIC: pytest.approx(.02)}, "suites": {"S1": {METRIC: .01}, "S2": {METRIC: .03}}}
    assert values["2"]["pooled"][METRIC] == pytest.approx(-.03)
    other = seed_values(pairs, [0, 1, 2], relative_to=1)
    assert other["0"]["pooled"][METRIC] == pytest.approx(-.02) and other["0"]["suites"]["S2"][METRIC] == -.03
    assert other["2"]["pooled"][METRIC] == pytest.approx(-.05)
    moved = recentred(values, 1)
    assert moved["0"]["pooled"][METRIC] == pytest.approx(-.02) and moved["2"]["suites"]["S1"][METRIC] == \
        pytest.approx(-.06)
    with pytest.raises(ValueError, match="no pair compares init seeds 0 and 3"):
        seed_values(pairs, [0, 1, 3], relative_to=0)


def test_the_baseline_seeds_mean_and_sd_per_metric():
    summary = baseline_summary(seed_values(calibration_pairs(), [0, 1, 2], relative_to=0))
    assert summary == {METRIC: {"mean": pytest.approx(-.01 / 3), "sd": pytest.approx(statistics.stdev([0, .02, -.03])),
                                "n": 3}}


def test_a_metric_no_suite_of_a_pair_compares_has_no_band_from_that_pair():
    pairs = [pair(comparison(None)), pair(comparison(entry(-.04)))]
    assert pairs[0] == {"pooled": {}, "suites": {}}
    assert null_band(pairs) == {METRIC: pytest.approx(.04)}


# Six baseline seeds' values of METRIC against the baseline screen (whose own seed is the first): mean −.01, SD .0141.
BASE = (0., -.02, -.01, -.03, .01, -.01)
SB = statistics.stdev(BASE)


def seeds(values=BASE, suites=4):
    """A calibration's baseline seeds (``noise.seed_values``), each with one value of METRIC, pooled and per suite."""
    return {str(seed): {"pooled": {METRIC: value}, "suites": {f"S{n}": {METRIC: value} for n in range(1, suites + 1)}}
            for seed, value in enumerate(values)}


CALIBRATED = object()


def decision(*replicates, baseline=CALIBRATED, band=.05):
    """``replicates``: per candidate init seed, its suites' entries, each against the baseline screen."""
    found = decide([comparison(*suites) for suites in replicates], seeds() if baseline is CALIBRATED else baseline,
                   [METRIC], band={METRIC: band})
    return found[METRIC]


def test_welch_compares_the_candidates_mean_with_the_baselines():
    candidate = [.03, .01, .02]
    sc = statistics.stdev(candidate)
    se = math.sqrt(sc ** 2 / 3 + SB ** 2 / 6)
    assert welch(candidate, BASE) == {"candidate_mean": pytest.approx(.02), "candidate_sd": pytest.approx(sc),
                                      "baseline_mean": pytest.approx(-.01), "baseline_sd": pytest.approx(SB), "k": 3,
                                      "n": 6, "diff": pytest.approx(.03), "se": pytest.approx(se),
                                      "z": pytest.approx(.03 / se)}


def test_one_candidate_seed_is_given_the_baselines_spread():
    """With one seed the candidate's spread is unmeasured; it is taken to be the baseline's (sc = sb)."""
    found = welch([.03], BASE)
    assert found["candidate_sd"] is None and found["se"] == pytest.approx(SB * math.sqrt(1 + 1 / 6))
    assert found["z"] == pytest.approx(.04 / (SB * math.sqrt(1 + 1 / 6)))


def test_welch_needs_a_candidate_value_and_two_baseline_values_and_has_no_z_without_spread():
    with pytest.raises(ValueError, match="two baseline"):
        welch([.01], [0.])
    with pytest.raises(ValueError, match="candidate"):
        welch([], BASE)
    assert welch([0.], [0., 0., 0.])["z"] is None


SE1 = SB * math.sqrt(1 + 1 / 6)                       # one candidate seed against BASE: its spread taken as BASE's


def test_better_needs_a_pooled_z_of_two_and_no_suite_regressing():
    """One candidate seed: its pooled Δ less the baseline seeds' mean, over Welch's standard error with the candidate's
    spread the baseline's; each suite likewise, its Δ against the baseline seeds' values in that suite. No suite is at
    z ≤ −2, so the gain is better beyond noise. S4's −.5 pp is z +0.3 there: the baseline seeds sit 1 pp below the
    baseline screen."""
    found = decision([entry(.06), entry(.04), entry(.03), entry(-.005)])
    assert (found["k"], found["n"], found["decision"], found["regressions"]) == (1, 6, "beyond noise (better)", [])
    assert found["pooled"] == pytest.approx(.03125) and found["candidate"] == [pytest.approx(.03125)]
    assert found["baseline_mean"] == pytest.approx(-.01) and found["baseline_sd"] == pytest.approx(SB)
    assert found["diff"] == pytest.approx(.04125) and found["z"] == pytest.approx(.04125 / SE1)
    assert found["band"] == .05                     # above the pooled Δ: the band is a reference, not the rule
    assert {suite: entry["z"] for suite, entry in found["suite_z"].items()} == \
        {"S1": pytest.approx(.07 / SE1), "S2": pytest.approx(.05 / SE1), "S3": pytest.approx(.04 / SE1),
         "S4": pytest.approx(.005 / SE1)}
    assert found["suite_z"]["S4"]["diff"] == pytest.approx(.005) and not found["suite_z"]["S4"]["limited"]
    assert decision([entry(.01)] * 4)["decision"] == "within noise"                         # diff .02: z 1.3


def test_a_gain_on_the_hard_suites_while_the_others_hold_is_better_whatever_their_signs():
    """S1 gains, the other suites sit at their baseline seeds' values: one suite of four agrees in sign (reported),
    and nothing regresses."""
    found = decision([entry(.15), entry(-.01), entry(-.01), entry(-.01)])
    assert (found["agree"], found["suites"], found["decision"]) == (1, 4, "beyond noise (better)")


def test_a_regressing_suite_makes_a_pooled_gain_mixed_and_is_named():
    found = decision([entry(.12), entry(.06), entry(.06), entry(-.05)])        # S4: diff −.04, z −2.6
    assert found["z"] >= 2 and found["suite_z"]["S4"]["z"] == pytest.approx(-.04 / SE1)
    assert (found["decision"], found["regressions"]) == ("mixed", ["S4"])


def test_a_suite_with_too_few_clusters_or_pairs_is_reported_but_never_counted_as_a_regression():
    for test in ("too_few_clusters", "too_few_pairs"):
        found = decision([entry(.12), entry(.06), entry(.06), entry(-.05, test=test)])
        limited = found["suite_z"]["S4"]
        assert limited["limited"] and limited["z"] == pytest.approx(-.04 / SE1)
        assert (found["decision"], found["regressions"]) == ("beyond noise (better)", [])
    # limited in any seed's comparison of the suite
    two = decision([entry(.12), entry(.06), entry(.06), entry(-.05)], [entry(.12), entry(.06), entry(.06),
                                                                        entry(-.05, test=FEW)])
    assert two["suite_z"]["S4"]["limited"] and two["regressions"] == []


def test_a_pooled_loss_is_worse_whatever_the_suites():
    assert decision([entry(-.06)] * 4)["decision"] == "beyond noise (worse)"                # diff −.05: z −3.3
    found = decision([entry(.10), entry(-.09), entry(-.09), entry(-.09)])                   # S1 gains alone
    assert found["z"] <= -2 and found["decision"] == "beyond noise (worse)"


def test_several_candidate_seeds_are_compared_by_their_mean_and_spread_pooled_and_per_suite():
    found = decision([entry(.02)] * 4, [entry(.03)] * 4, [entry(.04)] * 4)
    se = math.sqrt(.01 ** 2 / 3 + SB ** 2 / 6)
    assert found["candidate"] == [pytest.approx(.02), pytest.approx(.03), pytest.approx(.04)] and found["k"] == 3
    assert found["candidate_sd"] == pytest.approx(.01) and found["z"] == pytest.approx(.04 / se)
    assert found["decision"] == "beyond noise (better)"
    # S2 falls in every seed: its seeds' Δs (−.06, −.05, −.04) against the baseline seeds there give z −4.9
    mixed = decision([entry(.02), entry(-.06), entry(.10), entry(.10)],
                     [entry(.03), entry(-.05), entry(.10), entry(.10)],
                     [entry(.04), entry(-.04), entry(.10), entry(.10)])
    assert mixed["suite_z"]["S2"]["z"] == pytest.approx(-.04 / se) and mixed["suite_z"]["S2"]["k"] == 3
    assert (mixed["decision"], mixed["regressions"]) == ("mixed", ["S2"])


def test_a_suite_without_spread_or_two_baseline_values_has_no_z_and_never_regresses():
    flat = seeds()
    for found in flat.values():
        found["suites"]["S4"] = {METRIC: 0.}                    # every baseline seed at the baseline screen's value
    del flat["1"]["suites"]["S3"], flat["2"]["suites"]["S3"], flat["3"]["suites"]["S3"], flat["4"]["suites"]["S3"]
    del flat["5"]["suites"]["S3"]                                # only the baseline screen's own seed has S3
    found = decision([entry(.12), entry(.06), entry(-.30), entry(0.)], baseline=flat)
    assert found["suite_z"]["S4"]["z"] is None and found["suite_z"]["S3"]["z"] is None
    assert found["regressions"] == []


def test_the_signs_are_kept_as_information_over_the_suites_with_enough_clusters():
    """Enough clusters is the assay's own threshold for a verdict, in every seed's comparison of the suite."""
    clustered = decision([entry(.09), entry(.09), entry(-.03, test=FEW), entry(-.03, test=FEW)])
    assert (clustered["agree"], clustered["suites"], clustered["basis"]) == (2, 2, "clusters")
    fallback = decision([entry(.21, test=FEW), entry(-.03, test=FEW), entry(-.03, test=FEW), entry(-.03, test=FEW)])
    assert (fallback["agree"], fallback["suites"], fallback["basis"]) == (1, 4, "all")
    either = decision([entry(.09), entry(.09), entry(.09), entry(.09, test=FEW)],
                      [entry(.09), entry(.09), entry(.09, test=FEW), entry(.09, test=FEW)])
    assert (either["suites"], either["basis"]) == (2, "clusters")


def test_without_three_baseline_seeds_the_headline_is_uncalibrated():
    for baseline in (None, seeds(BASE[:2])):
        found = decision([entry(.5)] * 4, baseline=baseline)
        assert (found["decision"], found["z"], found["diff"], found["baseline_mean"]) == ("uncalibrated", None, None,
                                                                                           None)
        assert found["pooled"] == pytest.approx(.5) and (found["agree"], found["suites"]) == (4, 4)
    other = decide([comparison(entry(.5))], {str(seed): {"pooled": {"other.metric": 0.}, "suites": {}}
                                             for seed in range(3)}, [METRIC])
    assert other[METRIC]["decision"] == "uncalibrated" and other[METRIC]["band"] is None


def test_baseline_seeds_that_never_differ_bound_nothing():
    found = decision([entry(.5)] * 4, baseline=seeds((0., 0., 0.)))
    assert (found["z"], found["decision"]) == (None, "uncalibrated: spread 0")


def test_a_metric_no_suite_compares_has_no_decision():
    found = decision([None, None])
    assert (found["pooled"], found["candidate"], found["agree"], found["suites"], found["basis"], found["decision"]) \
        == (None, [], 0, 0, None, None)
