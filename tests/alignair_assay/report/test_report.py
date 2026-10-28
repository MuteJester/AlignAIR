
import pytest
import json
import math

from alignair_assay.report import build_comparison_report, build_score_report, headline_table, to_markdown
from alignair_assay.stats import MetricSummary, compare

pytestmark = pytest.mark.component


def _ms(mean, n=10, cov=1.0):
    return MetricSummary(mean=mean, n=n, n_applicable=n, coverage=cov)


def _summary():
    return {
        "zzz.other": _ms(0.5),
        "productive.correct": _ms(0.9),
        "v.mutation_count_err": _ms(1.23456),
        "clone_key.exact": _ms(0.973, n=20, cov=0.7),
        "junction.nt_exact": _ms(None, n=0, cov=0.0),
        "aaa.other": _ms(math.nan),
    }


def test_report_keys():
    rep = build_score_report("tool", _summary(), {"stratum": {"s1": {"clone_key.exact": _ms(0.5)}}}, {"seed": 1})
    assert rep["kind"] == "alignair_assay.score_report"
    assert rep["version"] == "1"
    assert rep["tool"] == "tool"
    assert set(rep) == {"kind", "version", "tool", "overall", "by_slice", "provenance"}
    assert rep["overall"]["aaa.other"]["mean"] is None
    assert rep["by_slice"]["stratum"]["s1"]["clone_key.exact"]["mean"] == 0.5
    json.dumps(rep, allow_nan=False)


def test_markdown_headline_order():
    md = to_markdown(build_score_report("tool", _summary(), {}, {}))
    rows = [ln for ln in md.splitlines() if ln.startswith("|")][2:]
    names = [r.split("|")[1].strip() for r in rows]
    assert names == ["clone_key.exact", "junction.nt_exact", "v.mutation_count_err", "productive.correct",
                     "aaa.other", "zzz.other"]


def test_true_allele_accuracy_sits_beside_top1_in_set():
    summary = {"v.top1_in_set": _ms(0.9), "v.top1_generative": _ms(0.8), "j.top1_in_set": _ms(0.99),
               "j.top1_generative": _ms(0.98), "d.top1_generative": _ms(0.7), "v.gene_top1": _ms(0.95)}
    md = to_markdown(build_score_report("tool", summary, {}, {}))
    names = [ln.split("|")[1].strip() for ln in md.splitlines() if ln.startswith("| ")][1:]
    assert names == ["v.top1_in_set", "v.top1_generative", "j.top1_in_set", "j.top1_generative",
                     "d.top1_generative", "v.gene_top1"]


def test_comparison_shows_informational_metrics_untested():
    rep = build_comparison_report("A", "B", {}, {}, {}, {}, informational={
        "v.top1_generative": {"a": _ms(0.9, n=20), "b": _ms(0.8, n=16, cov=0.8)}, "v.set_size": {"a": _ms(1.5),
                                                                                              "b": None}})
    assert rep["informational"]["v.top1_generative"]["b"] == {"mean": 0.8, "n": 16, "n_applicable": 16,
                                                              "coverage": 0.8}
    md = to_markdown(rep)
    section = md.split("## Informational (not compared)")[1]
    assert "| Metric | A | B |" in section
    assert "| v.top1_generative | 90.0% (n=20, 100%) | 80.0% (n=16, 80%) |" in section
    assert "| v.set_size | 1.500 (n=10, 100%) | — |" in section
    # Without informational metrics there is no such section.
    assert "Informational" not in to_markdown(build_comparison_report("A", "B", {}, {}, {}, {}))


def test_markdown_shows_coverage_and_formats():
    md = to_markdown(build_score_report("tool", _summary(), {}, {}))
    assert "| clone_key.exact | 97.3% | 20 | 70% |" in md
    assert "| v.mutation_count_err | 1.235 | 10 | 100% |" in md
    assert "| junction.nt_exact | — | 0 | 0% |" in md


def _comparison():
    # B covers 16 of 20 clone keys, so its own mean, the paired means and the difference all differ.
    a = {f"c{i}": {"clone_key.exact": float(i < 10), "v.mutation_count_err": 2.0} for i in range(20)}
    b = {f"c{i}": {"clone_key.exact": 1.0 if i < 16 else math.nan, "v.mutation_count_err": 1.0} for i in range(20)}
    comp = compare(a, b, ["clone_key.exact", "v.mutation_count_err", "zzz.other_err"],
                   lower_is_better={"v.mutation_count_err", "zzz.other_err"})
    sl = {"stratum": {"s1": {"clone_key.exact": _ms(0.5)}}}
    return build_comparison_report("A", "B", comp, sl, sl, {"x": 1})


def test_comparison_report_and_markdown():
    rep = _comparison()
    assert rep["kind"] == "alignair_assay.comparison_report"
    assert rep["a"] == "A" and rep["b"] == "B"
    json.dumps(rep, allow_nan=False)
    md = to_markdown(rep)
    assert "| Metric | A | B | n paired | A (paired) | B (paired) | Δ (b−a) | 95% CI | Verdict |" in md
    assert "Verdicts: Holm-corrected paired tests (exact McNemar for 0/1 metrics, paired z otherwise)" in md
    assert "differences in percentage points (pp)" in md
    # Own means with n and coverage, then the paired subset: rate differences in pp, errors as raw numbers.
    assert "| clone_key.exact | 50.0% (n=20, 100%) | 100.0% (n=16, 80%) | 16 | 62.5% | 100.0% | +37.5 pp | [+" in md
    cells = [c.strip() for c in next(ln for ln in md.splitlines() if ln.startswith("| clone_key.exact |")).split("|")]
    ci_low, ci_high = cells[8].strip("[]").split(", ")
    assert ci_low.endswith(" pp") and ci_high.endswith(" pp")
    assert "| v.mutation_count_err | 2.000 (n=20, 100%) | 1.000 (n=20, 100%) | 20 | 2.000 | 1.000 | -1.000 | [" in md


def test_headline_table_has_only_headline_rows():
    table = headline_table(_comparison())
    lines = table.splitlines()
    assert lines[0] == "| Metric | A | B | n paired | A (paired) | B (paired) | Δ (b−a) | 95% CI | Verdict |"
    assert [ln.split("|")[1].strip() for ln in lines[2:]] == ["clone_key.exact", "v.mutation_count_err"]


def test_headline_rows_are_a_parameter():
    from alignair_assay.core.headline import HEADLINE
    from alignair_assay.report import markdown

    # One headline list, owned by core; the markdown default is that list.
    assert markdown.HEADLINE is HEADLINE
    rep = _comparison()
    table = headline_table(rep, headline=("v.mutation_count_err", "zzz.other_err", "not.compared"))
    assert [ln.split("|")[1].strip() for ln in table.splitlines()[2:]] == ["v.mutation_count_err", "zzz.other_err"]
    # The overall table leads with the given rows; slice tables show only them.
    sl = {"stratum": {"s1": {"clone_key.exact": _ms(0.5), "v.mutation_count_err": _ms(1.0)}}}
    md = to_markdown(build_score_report("tool", _summary(), sl, {}), headline=("v.mutation_count_err",))
    names = [ln.split("|")[1].strip() for ln in md.split("## By stratum")[0].splitlines() if ln.startswith("| ")]
    assert names[1] == "v.mutation_count_err"
    by_stratum = md.split("## By stratum")[1]
    assert "| v.mutation_count_err |" in by_stratum and "| clone_key.exact |" not in by_stratum


def test_slice_cells_show_coverage():
    sl = {"stratum": {"s1": {"clone_key.exact": _ms(0.973, cov=0.7), "junction.nt_exact": _ms(None, n=0, cov=0.0)}}}
    md = to_markdown(build_score_report("tool", _summary(), sl, {}))
    assert "| clone_key.exact | 97.3% (n=10, 70%) |" in md
    assert "| junction.nt_exact | — |" in md


def test_comparison_keeps_b_only_dimension_and_coverage():
    a = {"c0": {"clone_key.exact": 1.0}, "c1": {"clone_key.exact": 0.0}}
    b = {"c0": {"clone_key.exact": 1.0}, "c1": {"clone_key.exact": 1.0}}
    comp = compare(a, b, ["clone_key.exact"], lower_is_better=set())
    rep = build_comparison_report("A", "B", comp, {"stratum": {"s1": {"clone_key.exact": _ms(0.5, cov=0.7)}}},
                                  {"d_len_bin": {"long": {"clone_key.exact": _ms(0.9)}}}, {})
    md = to_markdown(rep)
    assert "## By d_len_bin (A / B)" in md
    assert "— / 90.0% (n=10, 100%)" in md
    assert "50.0% (n=10, 70%) / —" in md


def test_pipes_escaped_and_missing_verdict():
    rep = build_score_report("t|x", {"a|b": _ms(0.5)}, {"dim": {"l|1": {"clone_key.exact": _ms(0.5)}}}, {})
    md = to_markdown(rep)
    assert "| a\\|b | 50.0% |" in md
    assert "l\\|1" in md
    comp = {"clone_key.exact": {"a": None, "b": None, "n_paired": 0, "mean_a_paired": None, "mean_b_paired": None,
                                "delta": None, "ci95": None, "verdict": None, "coverage_a": 0.0, "coverage_b": 0.0}}
    md = to_markdown(build_comparison_report("A", "B", comp, {}, {}, {}))
    assert "| clone_key.exact | — | — | 0 | — | — | — | — | — |" in md


def test_the_alignair_code_two_checkpoints_ran_under():
    from alignair_assay.report.markdown import alignair_code_text

    one, other = "a" * 64, "b" * 64
    assert alignair_code_text({"x@a": one, "x@b": one}) == f"both ran under AlignAIR code `{'a' * 12}`"
    assert alignair_code_text({"x@a": one, "x@b": other}) == (
        f"different AlignAIR code: x@a `{'a' * 12}`, x@b `{'b' * 12}`")
    assert alignair_code_text({"x@a": one, "x@b": None}) == "AlignAIR code not recorded"
