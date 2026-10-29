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


def test_twelve_suites_fit_one_screen():
    text = render_headline(comparison(), title=["Screen wide: lab-wide-v1 at 20000 steps."])
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


def test_a_different_alignair_code_or_the_same_model_is_flagged():
    found = comparison()
    found["code"] = {"this": "c" * 64, "baseline": "d" * 64, "same": False}
    assert "different AlignAIR code" in render_headline(found)
    found["same_model"] = True
    assert "same model" in render_headline(found)
