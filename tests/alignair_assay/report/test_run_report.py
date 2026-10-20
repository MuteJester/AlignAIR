"""The run report: markdown rendered from the plain summary dict ``workflow.summary.collect_run_summary`` builds."""
import copy

from alignair_assay.report.run import run_report_markdown


def _entry(a, b, verdict):
    return {"a": {"mean": a, "n": 20, "n_applicable": 20, "coverage": 1.0},
            "b": {"mean": b, "n": 20, "n_applicable": 20, "coverage": 1.0}, "n_paired": 20,
            "mean_a_paired": a, "mean_b_paired": b, "delta": b - a, "ci95": [b - a - 0.1, b - a + 0.1],
            "verdict": verdict}


def _cell(mean, n=10, coverage=1.0):
    return None if mean is None else {"mean": mean, "n": n, "n_applicable": n, "coverage": coverage}


def _summary() -> dict:
    rows = {"junction.nt_exact": _entry(0.95, 0.70, "a_better"), "v.top1_in_set": _entry(0.90, 0.95, "b_better"),
            "j.top1_in_set": _entry(0.99, 0.99, "tie_or_inconclusive")}
    s1 = {"name": "S1", "title": "Core repertoire", "purpose": "Measures annotation accuracy.",
          "primary": ["junction.nt_exact", "v.top1_in_set", "j.top1_in_set"], "cases": 5000, "excluded": 2,
          "truth": {"excluded": 2, "warnings": 0, "unscorable": {"v": 0, "d": 0, "j": 0},
                    "recalled": {"v": 160, "d": 179, "j": 2}},
          "oracle": {"status": "perfect", "cases": 4998, "problems": 0, "examples": []},
          "arms": {"shared": {"compared": True, "link": "scores/S1/shared/comparison.md", "rows": rows,
                              "verdicts": {"a_better": 1, "b_better": 1, "tie_or_inconclusive": 1}}}}
    s8 = {"name": "S8", "title": "Genotype", "purpose": "Measures genotype use.", "primary": ["v.top1_in_set"],
          "cases": 3000, "excluded": 0,
          "truth": {"excluded": 0, "warnings": 3, "unscorable": {"v": 0, "d": 2, "j": 0},
                    "recalled": {"v": 40, "d": None, "j": 0}},
          "oracle": {"status": "imperfect", "cases": 3000, "problems": 4,
                     "examples": [["S8:cohort:000001", "v.top1_in_set", 0.0]]},
          "arms": {"shared": {"compared": True, "link": "scores/S8/shared/comparison.md",
                              "rows": {"v.top1_in_set": _entry(0.9, 0.8, "a_better")},
                              "verdicts": {"a_better": 1, "b_better": 0, "tie_or_inconclusive": 0}},
                   "genotype": {"compared": False, "link": None, "rows": {}, "verdicts": None}}}
    return {
        "kind": "alignair_assay.run_report", "version": "2", "a": "igblast", "b": "alignair",
        "config": {"suites": ["S1", "S8", "S10", "S11"], "tools": ["igblast", "alignair"], "split": "test",
                   "reference": "HUMAN_IGH_OGRDB", "seed": 20261019, "scale": 1.0},
        "config_hash": "0123456789abcdef" * 4,
        "failed_runs": [{"tool": "alignair", "suite": "S8", "arm": "genotype", "status": "failed", "returncode": 1,
                         "failed_donor": "donor00", "error": "AttributeError: 'tuple' object has no attribute 'items'",
                         "run_json": "runs/alignair/S8/genotype/run.json"}],
        "oracle_failures": [{"suite": "S8", "status": "imperfect", "problems": 4,
                             "examples": [["S8:cohort:000001", "v.top1_in_set", 0.0]]},
                            {"suite": "S10", "status": "not scored", "cases": 0, "problems": 0, "examples": [],
                             "reason": "the oracle run is missing: no run record: the run never finished"}],
        "truth_warnings": [{"suite": "S8", "unscorable": {"v": 0, "d": 2, "j": 0}, "warnings": 3,
                            "build_json": "suites/S8/build.json"}],
        "suites": [s1, s8],
        "s10_vs_s1": {"baseline": "S1 full_igg", "settings": ["uniform", "np_shift"], "tools": {
            "igblast": [{"metric": "junction.nt_exact", "s1": _cell(0.98), "s10": _cell(0.93, 40), "delta": -0.05,
                         "by_setting": {"uniform": _cell(0.95), "np_shift": _cell(0.91, 10, 0.5)}}],
            "alignair": [{"metric": "junction.nt_exact", "s1": _cell(0.80), "s10": None, "delta": None,
                          "by_setting": {"uniform": None, "np_shift": None}}]}},
        "s11_parity": {"alpha": 0.05, "tools": {
            "igblast": {"pairs": 1000, "family_size": 60, "differ": [], "rows": [
                {"metric": "orientation.correct", "forward": 1.0, "reverse": 1.0, "delta": 0.0, "n_paired": 1000,
                 "verdict": "tie_or_inconclusive"}]},
            "alignair": {"pairs": 1000, "family_size": 60, "differ": ["junction.nt_exact", "v.top1_in_set"], "rows": [
                {"metric": "junction.nt_exact", "forward": 0.8, "reverse": 0.7, "delta": -0.1, "n_paired": 1000,
                 "verdict": "a_better"}]}}},
        "genotype_arms": [{"suite": "S8", "arms": ["shared", "genotype"], "rows": [
            {"metric": "v.top1_in_set", "values": {"igblast": {"shared": _cell(0.9), "genotype": _cell(0.95)},
                                                   "alignair": {"shared": _cell(0.8, 1, 0.1), "genotype": None}}}]}],
        "excess_alleles": [{"suite": "S8", "rows": [
            {"tool": "oracle", "arm": "shared", "table": {
                "min_reads": 3,
                "v": {"n": 120, "excess": 0, "fraction": 0.0, "avoidable_excess": 0, "avoidable_fraction": 0.0},
                "j": {"n": 40, "excess": 0, "fraction": 0.0, "avoidable_excess": 0, "avoidable_fraction": 0.0}}},
            {"tool": "igblast", "arm": "shared", "table": {
                "min_reads": 3,
                "v": {"n": 118, "excess": 9, "fraction": 9 / 118, "avoidable_excess": 3,
                      "avoidable_fraction": 3 / 118},
                "j": {"n": 0, "excess": 0, "fraction": None, "avoidable_excess": 0, "avoidable_fraction": None}}},
            {"tool": "alignair", "arm": "genotype", "table": None}]}],
        "speed": {"host": {"cpu": "Test CPU", "logical_cpus": 8}, "n_tiny": 50, "n_full": 4000, "repeats": 3,
                  "results": {"igblast_8t": {"cold_start_s": 0.5, "steady_reads_per_s": 250.0,
                                             "median_full_s": 16.5, "peak_rss_kb": 126000}}},
    }


def _sections(md: str) -> list[str]:
    return [line for line in md.splitlines() if line.startswith("## ")]


def test_report_renders_sections_from_summary():
    md = run_report_markdown(_summary())
    assert md.startswith("# AlignAIR assay run: igblast vs alignair\n")
    assert "split test, seed 20261019, scale 1.0, reference HUMAN_IGH_OGRDB; config hash 0123456789ab" in md
    assert _sections(md) == ["## Failed runs", "## Oracle check failures", "## Truth warnings", "## Summary",
                             "## Truth", "## Suites", "## S10 vs S1 (held-out settings)", "## S11 orientation parity",
                             "## Genotype arms", "## Excess alleles", "## Speed"]

    failed = md[md.index("## Failed runs"):md.index("## Oracle check failures")]
    assert ("| alignair | S8 | genotype | failed | 1 | donor00 | AttributeError: 'tuple' object has no attribute "
            "'items' | runs/alignair/S8/genotype/run.json |") in failed
    assert "not scored" in failed
    assert "| S8 | imperfect | 4 | S8:cohort:000001 v.top1_in_set = 0.0 |" in md
    # An oracle that was not scored leaves the truth unchecked: it is named with its reason, not as a defect.
    oracle = md[md.index("## Oracle check failures"):md.index("## Truth warnings")]
    assert "| S10 | not scored | 0 | the oracle run is missing: no run record: the run never finished |" in oracle
    assert "imperfect" in oracle and "truth defect" in oracle and "unchecked" in oracle

    # A suite with unscorable segments is flagged: they are not scored.
    flagged = md[md.index("## Truth warnings"):md.index("## Summary")]
    assert "| S8 | 0 | 2 | 0 | 3 |" in flagged and "suites/S8/build.json" in flagged and "S1" not in flagged
    truth = md[md.index("## Truth\n"):md.index("## Suites")]
    assert ("| Suite | Cases | Excluded | Cases with warnings | Unscorable V | Unscorable D | Unscorable J | "
            "Re-called V | Re-called D | Re-called J |") in truth
    assert "| S1 | 5000 | 2 | 0 | 0 | 0 | 0 | 160 | 179 | 2 |" in truth
    assert "| S8 | 3000 | 0 | 3 | 0 | 2 | 0 | 40 | — | 0 |" in truth

    summary = md[md.index("## Summary"):md.index("## Suites")]
    assert ("| Suite | Arm | Cases | Excluded | Oracle | igblast better | alignair better | Tie or inconclusive |"
            in summary)
    assert "| S1 | shared | 5000 | 2 | perfect | 1 | 1 | 1 |" in summary
    assert "| S8 | genotype | 3000 | 0 | imperfect (4 problems) | not compared | — | — |" in summary

    suites = md[md.index("## Suites"):md.index("## S10 vs S1")]
    assert "### S1: Core repertoire" in suites and "Measures annotation accuracy." in suites
    assert "[comparison](scores/S1/shared/comparison.md)" in suites
    # Rows in the suite's primary order, with the comparison table's columns.
    assert suites.index("| junction.nt_exact |") < suites.index("| v.top1_in_set |") < suites.index(
        "| j.top1_in_set |")
    assert "| junction.nt_exact | 95.0% (n=20, 100%) | 70.0% (n=20, 100%) | 20 |" in suites
    assert "Not compared" in suites[suites.index("### S8"):]

    s10 = md[md.index("## S10 vs S1"):md.index("## S11")]
    assert "unpaired" in s10 and "informational" in s10 and "S1 full_igg" in s10
    # Every mean carries its n and coverage.
    assert ("| junction.nt_exact | 98.0% (n=10, 100%) | 93.0% (n=40, 100%) | -5.0 pp | 95.0% (n=10, 100%) "
            "| 91.0% (n=10, 50%) |") in s10
    assert "| junction.nt_exact | 80.0% (n=10, 100%) | — | — | — | — |" in s10

    s11 = md[md.index("## S11"):md.index("## Genotype arms")]
    assert "**igblast** (1000 pairs): no metric differs" in s11
    assert "**alignair** (1000 pairs): differs on 2 of 60 metrics: junction.nt_exact, v.top1_in_set" in s11
    assert "| junction.nt_exact | 80.0% | 70.0% | -10.0 pp | 1000 | forward better |" in s11

    genotype = md[md.index("## Genotype arms"):md.index("## Speed")]
    assert "| Metric | igblast shared | igblast genotype | alignair shared | alignair genotype |" in genotype
    assert "| v.top1_in_set | 90.0% (n=10, 100%) | 95.0% (n=10, 100%) | 80.0% (n=1, 10%) | — |" in genotype

    excess = md[md.index("## Excess alleles"):md.index("## Speed")]
    assert "at least 3 reads" in excess and "avoidable" in excess
    assert "| Suite | Tool | Arm | V, all calls | V, avoidable | J, all calls | J, avoidable |" in excess
    # Every fraction shows its n; a table over no donor gene shows n=0, a run without a table a dash.
    assert "| S8 | oracle | shared | 0.0% (n=120) | 0.0% (n=120) | 0.0% (n=40) | 0.0% (n=40) |" in excess
    assert "| S8 | igblast | shared | 7.6% (n=118) | 2.5% (n=118) | — (n=0) | — (n=0) |" in excess
    assert "| S8 | alignair | genotype | — | — | — | — |" in excess

    speed = md[md.index("## Speed"):]
    assert "| igblast_8t | 0.50 | 250.0 | 16.50 | 123.0 |" in speed and "container start" in speed


def test_sections_without_data_are_left_out():
    summary = _summary()
    for key in ("s10_vs_s1", "s11_parity", "speed"):
        summary[key] = None
    summary.update(failed_runs=[], oracle_failures=[], truth_warnings=[], genotype_arms=[], excess_alleles=[])
    assert _sections(run_report_markdown(summary)) == ["## Summary", "## Truth", "## Suites"]
    for suite in summary["suites"]:
        suite["truth"] = None  # a build record without a truth summary
    assert _sections(run_report_markdown(summary)) == ["## Summary", "## Suites"]

    original = copy.deepcopy(summary)
    run_report_markdown(summary)
    assert summary == original  # rendering does not change the summary


def test_clustered_comparisons_show_their_clusters():
    summary = _summary()
    rows = summary["suites"][0]["arms"]["shared"]["rows"]
    rows["junction.nt_exact"].update(n_clusters=4, test="too_few_clusters", verdict="tie_or_inconclusive")
    md = run_report_markdown(summary)
    suites = md[md.index("## Suites"):md.index("## S10 vs S1")]
    assert "| junction.nt_exact | 95.0% (n=20, 100%) | 70.0% (n=20, 100%) | 20 (4 clusters) |" in suites
    assert "| tie_or_inconclusive (too few clusters) |" in suites
    assert suites.count("fewer than 10 clusters") == 1
    assert "clusters" not in run_report_markdown(_summary())
