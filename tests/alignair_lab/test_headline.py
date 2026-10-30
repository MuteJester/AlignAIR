"""A screen's headline fits one screen for twelve suites and shows, per suite, each headline metric's Δ, CI, verdict
and reads fixed / broken, and each boundary's Δ, verdict and reads fixed / broken under both decoders (their CIs are
in the assay's suite detail, as on its own summary page)."""
import pytest

from alignair_lab.headline import BOUNDARIES, DECODERS, HEADLINE, render_headline

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


# A null band per headline metric (Δs, as fractions): clone_key.exact and V top-1 are beyond it, the others within.
BAND = {"clone_key.exact": .01, "junction.nt_exact": .05, "v.top1_in_set": .005, "d.top1_in_set": .02,
        "j.top1_in_set": .03}


def test_twelve_suites_fit_one_screen():
    text = render_headline(comparison(), title=["Screen wide: lab-wide-v1 at 20000 steps."], null_band=BAND)
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


def test_each_headline_metric_weighs_its_pooled_delta_against_the_null_band():
    """The pooled Δ is read-weighted over the suites that compare the metric (clone_key.exact: S1 +12.3 and eleven
    suites +1.2, 500 reads each), set beside the null band and the suites, of those with a CI, whose Δ has its sign;
    beyond noise only above the band with at least 2/3 of the signs."""
    found = rows(render_headline(comparison(), null_band=BAND))
    assert cells(found["pooled"]) == ["+2.1 / 1.0, 12/12 agree", "+1.2 / 5.0, 11/11 agree",
                                             "+1.2 / 0.5, 12/12 agree", "+1.2 / 2.0, 11/11 agree",
                                             "+1.3 / 3.0, 12/12 agree"]
    assert cells(found["noise"]) == ["beyond noise", "within noise", "beyond noise", "within noise", "within noise"]


def test_without_a_null_band_the_headline_is_uncalibrated_and_marks_nothing():
    found = rows(render_headline(comparison()))
    assert cells(found["pooled"])[0] == "+2.1 / none, 12/12 agree"
    assert cells(found["noise"]) == ["uncalibrated"] * 5


def test_a_different_alignair_code_or_the_same_model_is_flagged():
    found = comparison()
    found["code"] = {"this": "c" * 64, "baseline": "d" * 64, "same": False}
    assert "different AlignAIR code" in render_headline(found)
    found["same_model"] = True
    assert "same model" in render_headline(found)
