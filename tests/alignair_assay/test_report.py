import json
import math

from alignair_assay.report import build_comparison_report, build_score_report, to_markdown
from alignair_assay.stats import MetricSummary, compare


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


def test_markdown_shows_coverage_and_formats():
    md = to_markdown(build_score_report("tool", _summary(), {}, {}))
    assert "| clone_key.exact | 97.3% | 20 | 70% |" in md
    assert "| v.mutation_count_err | 1.235 | 10 | 100% |" in md
    assert "| junction.nt_exact | — | 0 | 0% |" in md


def test_comparison_report_and_markdown():
    a = {f"c{i}": {"clone_key.exact": float(i % 2), "v.mutation_count_err": 2.0} for i in range(20)}
    b = {f"c{i}": {"clone_key.exact": 1.0, "v.mutation_count_err": 1.0} for i in range(20)}
    comp = compare(a, b, ["clone_key.exact", "v.mutation_count_err"], n_boot=200,
                   lower_is_better={"v.mutation_count_err"})
    sl = {"stratum": {"s1": {"clone_key.exact": _ms(0.5)}}}
    rep = build_comparison_report("A", "B", comp, sl, sl, {"x": 1})
    assert rep["kind"] == "alignair_assay.comparison_report"
    assert rep["a"] == "A" and rep["b"] == "B"
    json.dumps(rep, allow_nan=False)
    md = to_markdown(rep)
    assert "| Metric | A | B | Δ (b−a) | 95% CI | Verdict | Coverage A | Coverage B |" in md
    assert "Verdicts: Holm-corrected paired bootstrap; Δ and CI are raw (b − a)." in md
    assert "| clone_key.exact | 50.0% | 100.0% |" in md
