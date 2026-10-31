"""A screen's headline fits one screen for twelve suites and shows, per suite, each headline metric's Δ, CI, verdict
and reads fixed / broken, and each boundary's Δ, verdict and reads fixed / broken under both decoders (their CIs are
in the assay's suite detail, as on its own summary page)."""
import pytest

from alignair_lab.headline import (
    BOUNDARIES,
    DECODERS,
    HEADLINE,
    POOLED_HEADING,
    SEEDS_HEADING,
    SUITES_HEADING,
    render_headline,
    render_pooled_headline,
)

pytestmark = pytest.mark.component


def entry(delta, ci=(-0.012, 0.031), verdict="tie_or_inconclusive", fixed=12, broken=3, n_paired=500, test="paired"):
    return {"family": "baseline", "mean_baseline": 0.7, "mean_this": 0.7 + delta, "delta": delta, "ci95": list(ci),
            "verdict": verdict, "test": test, "p_value": 0.2, "n_paired": n_paired, "n_clusters": None,
            "fixed": fixed, "broken": broken}


def comparison(suites=12):
    headline = {metric: entry(0.012) for metric, _ in HEADLINE}
    headline.update({f"{metric}.{decoder}": entry(-0.005, verdict="a_better" if decoder == "soft" else "b_better")
                     for metric, _ in BOUNDARIES for decoder in ("soft", "argmax")})
    found = {f"S{n}": {"cases": 500, "clusters": 0, "headline": dict(headline)} for n in range(1, suites + 1)}
    found["S1"]["headline"]["clone_key.exact"] = entry(0.123, ci=(0.101, 0.145), verdict="b_better", fixed=70,
                                                       broken=8)
    found["S2"]["headline"]["d.top1_in_set"] = entry(0.0, n_paired=0)
    del found["S3"]["headline"]["junction.nt_exact"]
    found["S9"]["headline"]["j.top1_in_set"] = entry(0.02, test="too_few_clusters")
    return {"baseline": "a" * 12 + "-" + "b" * 12, "source": "/base/model.step20000.alignair", "same_model": False,
            "replay_verified": True, "code": {"this": "c" * 64, "baseline": "c" * 64, "same": True},
            "suites": found, "totals": {"baseline": {"fixed": 1, "broken": 2}}}


# A null band per headline metric (Δs, as fractions), printed as a reference.
BAND = {"clone_key.exact": .01, "junction.nt_exact": .05, "v.top1_in_set": .005, "d.top1_in_set": .02,
        "j.top1_in_set": .03}
# Six baseline seeds' values against the baseline screen, the same on every metric and suite: mean −1.0, SD 1.4 pp.
SEEDS = (0., -.02, -.01, -.03, .01, -.01)


def calibration(values=SEEDS, band=BAND):
    """A calibration as a screen's headline takes it: the band and each baseline seed's values
    (``noise.seed_values``)."""
    metrics = [metric for metric, _ in HEADLINE]
    seeds = {str(seed): {"pooled": dict.fromkeys(metrics, value),
                         "suites": {f"S{n}": dict.fromkeys(metrics, value) for n in range(1, 13)}}
             for seed, value in enumerate(values)}
    return {"band": band, "seeds": seeds}


def test_twelve_suites_fit_one_screen():
    text = render_headline(comparison(), title=["Screen wide: lab-wide-v1 at 20000 steps."], calibration=calibration())
    lines = text.splitlines()
    assert lines[0] == "Screen wide: lab-wide-v1 at 20000 steps."
    assert len(lines) <= 55 and max(map(len, lines)) <= 132, (len(lines), max(map(len, lines)))


def test_each_headline_metric_shows_its_delta_ci_verdict_and_reads_fixed_and_broken():
    text = render_headline(comparison())
    first = text[:text.index("soft-argmax")]                      # the headline metrics' table
    rows = {line.split()[0]: line for line in first.splitlines() if line.startswith("S")}
    assert "+12.3 [10.1,14.5] + 70/8" in rows["S1"]
    assert "+1.2 [-1.2,3.1] = 12/3" in rows["S1"]
    assert "not paired" in rows["S2"]
    assert "+2.0 [-1.2,3.1] ? 12/3" in rows["S9"]                 # too few clusters to test
    assert "n/a" in rows["S3"]                                    # a metric the suite does not compare


def test_boundaries_show_both_decoders_and_each_column_tallies_its_verdicts():
    text = render_headline(comparison())
    soft, argmax = (text.index(heading) for _, heading in DECODERS)
    assert "soft-argmax" in DECODERS[0][1] and "soft" not in DECODERS[1][1] and soft < argmax
    soft_rows = [line for line in text[soft:argmax].splitlines() if line.startswith("S1 ")]
    argmax_rows = [line for line in text[argmax:].splitlines() if line.startswith("S1 ")]
    assert soft_rows[0].count("-0.5 - 12/3") == 6
    assert argmax_rows[0].count("-0.5 + 12/3") == 6
    tallies = [line for line in text.splitlines() if line.startswith("+/-/=")]
    assert len(tallies) == 3
    assert tallies[1].split()[1:] == ["0/12/0"] * 6 and tallies[2].split()[1:] == ["12/0/0"] * 6


def test_the_headline_says_its_verdicts_leave_out_training_noise():
    """Two GPU runs of one architecture on the same records differ at 20k steps by more than the read CIs."""
    text = render_headline(comparison())
    assert "run-to-run training noise" in text and "lab-null-v1" in text


def rows(text):
    """The headline metrics' table, by row label."""
    table = text[:text.index("soft-argmax")]
    return {line.split("  ")[0].strip(): line for line in table.splitlines()}


def cells(row):
    return [cell.strip() for cell in row.split("  ")[1:] if cell.strip()]


def test_each_headline_metric_weighs_its_pooled_delta_against_the_baseline_seeds():
    """The pooled Δ is read-weighted over the suites that compare the metric (clone_key.exact: S1 +12.3 and eleven
    suites +1.2, 500 reads each), beside the null band as a reference. The baseline seeds' values against the same
    baseline screen (mean −1.0, SD 1.4, n 6) give the diff (+3.1 for clone_key.exact) and Welch's z with the candidate's
    spread the baseline's, with the suites, of those with enough clusters for a verdict (J top-1: not S9), whose diff
    has its sign (information); better beyond noise at z ≥ 2 with no suite regressing."""
    found = rows(render_headline(comparison(), calibration=calibration()))
    assert cells(found["pooled"]) == ["+2.1 / 1.0", "+1.2 / 5.0", "+1.2 / 0.5", "+1.2 / 2.0", "+1.3 / 3.0"]
    assert cells(found["seeds"]) == ["-1.0±1.4 n6"] * 5
    assert cells(found["diff"]) == ["+3.1 z+2.0 12/12", "+2.2 z+1.4 11/11", "+2.2 z+1.4 12/12", "+2.2 z+1.4 11/11",
                                    "+2.3 z+1.5 11/11"]
    assert cells(found["noise"]) == ["beyond noise (better)"] + ["within noise"] * 4


def test_a_pooled_gain_with_a_regressing_suite_is_mixed_and_names_it():
    """S1 gains 30 pp on clone_key.exact while S5 loses 8: the pooled z is 2.5, S5's own z −4.6."""
    found = comparison()
    found["suites"]["S1"]["headline"]["clone_key.exact"] = entry(0.30)
    found["suites"]["S5"]["headline"]["clone_key.exact"] = entry(-0.08)
    assert cells(rows(render_headline(found, calibration=calibration()))["noise"])[0] == "mixed: S5"


def test_signs_counted_over_every_suite_for_want_of_clusters_are_marked():
    found = comparison()
    for suite in found["suites"].values():
        suite["headline"]["v.top1_in_set"] = entry(0.012, test="too_few_clusters")
    diff = cells(rows(render_headline(found, calibration=calibration()))["diff"])
    assert diff[2] == "+2.2 z+1.4 12/12*" and not diff[0].endswith("*")


def test_baseline_seeds_that_never_differ_mark_nothing():
    found = rows(render_headline(comparison(), calibration=calibration(values=(0., 0., 0.))))
    assert cells(found["seeds"])[0] == "+0.0±0.0 n3" and cells(found["diff"])[0] == "+2.1 z n/a 12/12"
    assert cells(found["noise"]) == ["uncalibrated: spread 0"] * 5


def test_without_three_baseline_seeds_the_headline_is_uncalibrated_and_marks_nothing():
    for given, seeds in ((None, "none"), (calibration(values=(0., -.02)), "n2 < 3")):
        found = rows(render_headline(comparison(), calibration=given))
        assert cells(found["pooled"])[0] == ("+2.1 / none" if given is None else "+2.1 / 1.0")
        assert cells(found["seeds"]) == [seeds] * 5 and cells(found["diff"])[0] == "n/a 12/12"
        assert cells(found["noise"]) == ["uncalibrated"] * 5


def test_a_different_alignair_code_or_the_same_model_is_flagged():
    found = comparison()
    found["code"] = {"this": "c" * 64, "baseline": "d" * 64, "same": False}
    assert "different AlignAIR code" in render_headline(found)
    found["same_model"] = True
    assert "same model" in render_headline(found)


def replicates(shifts=(0., .01, .02)):
    """Init seed -> its screen's comparison: ``comparison()`` with every headline Δ raised by the seed's shift."""
    found = {}
    for seed, shift in enumerate(shifts):
        summary = comparison()
        for suite in summary["suites"].values():          # the suites share their entries: each is replaced
            for metric, _ in HEADLINE:
                if metric in suite["headline"]:
                    shared = suite["headline"][metric]
                    suite["headline"][metric] = {**shared, "delta": shared["delta"] + shift}
        found[seed] = summary
    return found


def pooled_rows(text, heading):
    """The rows, by their first cell, of the table after ``heading``, up to the next blank line."""
    lines = text[text.index(heading):].splitlines()[1:]
    table = lines[:lines.index("")] if "" in lines else lines
    return {line.split("  ")[0].strip(): cells(line) for line in table}


def test_the_pooled_headline_sets_the_candidates_seeds_against_the_baseline_seeds():
    """Three seeds of clone_key.exact pool to +2.1, +3.1 and +4.1 (mean +3.1, SD 1.0). Against the six baseline seeds
    (mean −1.0, SD 1.4) the diff is +4.1, and Welch's z = 4.1 / sqrt(1.0²/3 + 1.4²/6) = 5.1."""
    text = render_pooled_headline(replicates(), calibration=calibration(), title=["Replicates wide: lab-wide-v1."],
                                  boundaries=(0, "wide"))
    assert text.splitlines()[0] == "Replicates wide: lab-wide-v1."
    table = pooled_rows(text, POOLED_HEADING)
    assert table["Metric"] == ["candidate", "baseline seeds", "diff", "z", "agree", "band", "noise"]
    better = "beyond noise (better)"
    assert table["clone_key.exact"] == ["+3.1±1.0 k3", "-1.0±1.4 n6", "+4.1", "+5.1", "12/12", "1.0", better]
    assert table["J top-1"][4:] == ["11/11", "3.0", better]                             # S9: too few clusters
    seeds = pooled_rows(text, SEEDS_HEADING)
    assert seeds["Metric"] == ["init 0", "init 1", "init 2"] and seeds["clone_key.exact"] == ["+2.1", "+3.1", "+4.1"]
    # each suite's z: its seeds' Δs against the baseline seeds there; S9's J top-1 has too few clusters (~)
    suites = pooled_rows(text, SUITES_HEADING)
    assert suites["Suite"] == [label for _, label in HEADLINE]
    assert suites["S1"][0] == "+17.5" and suites["S2"][0] == "+3.9" and suites["S9"][4] == "+4.9~"
    assert suites["S3"][1] == "-" and suites["S2"][3] == "-"                # S3 does not compare it; S2: not paired
    heading = next(line for line in text.splitlines() if "init seed 0" in line and "wide" in line)
    assert "alone" in heading and text.index(heading) < text.index(DECODERS[0][1]) < text.index(DECODERS[1][1])
    assert max(map(len, text.splitlines())) <= 132


def test_the_pooled_headline_names_the_suites_a_mixed_gain_regresses_on():
    found = replicates()
    for summary in found.values():
        summary["suites"]["S5"]["headline"]["clone_key.exact"] = entry(-0.08)
    text = render_pooled_headline(found, calibration=calibration(), boundaries=(0, "wide"))
    assert pooled_rows(text, POOLED_HEADING)["clone_key.exact"][-1] == "mixed: S5"
    assert pooled_rows(text, SUITES_HEADING)["S5"][0] == "-12.1!"


def test_one_seed_is_given_the_baselines_spread_and_no_calibration_marks_nothing():
    one = pooled_rows(render_pooled_headline(replicates((0.,)), calibration=calibration(), boundaries=(0, "wide")),
                      POOLED_HEADING)
    assert one["clone_key.exact"][:4] == ["+2.1 k1", "-1.0±1.4 n6", "+3.1", "+2.0"]
    none_text = render_pooled_headline(replicates(), calibration=None, boundaries=(0, "wide"))
    none = pooled_rows(none_text, POOLED_HEADING)
    assert none["clone_key.exact"] == ["+3.1±1.0 k3", "none", "n/a", "n/a", "12/12", "none", "uncalibrated"]
    assert SUITES_HEADING not in none_text and "Per-suite z: none, uncalibrated." in none_text
