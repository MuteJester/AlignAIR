from pathlib import Path

import pytest

from alignair_assay.core.layout import RunLayout


def test_layout_paths(tmp_path):
    layout = RunLayout(tmp_path)
    expected = [
        (layout.manifest, "manifest.json"),
        (layout.reference_dir, "reference"),
        (layout.reference_fasta, "reference/germline.fasta"),
        (layout.reference_json, "reference/reference.json"),
        (layout.suite_dir("S1"), "suites/S1"),
        (layout.cases("S1"), "suites/S1/cases.jsonl.gz"),
        (layout.truth("S1"), "suites/S1/truth.jsonl.gz"),
        (layout.reads("S1"), "suites/S1/reads.fasta"),
        (layout.donors("S1"), "suites/S1/donors.json"),
        (layout.build_json("S1"), "suites/S1/build.json"),
        (layout.run_dir("igblast", "S1", "shared"), "runs/igblast/S1/shared"),
        (layout.predictions("igblast", "S1", "shared"), "runs/igblast/S1/shared/predictions.tsv"),
        (layout.run_json("alignair-local", "S8", "genotype"), "runs/alignair-local/S8/genotype/run.json"),
        (layout.logs_dir("oracle", "S0", "shared"), "runs/oracle/S0/shared/logs"),
        (layout.work_dir("igblast"), "runs/igblast/_work"),
        (layout.donor_reads("alignair", "S1", "genotype", "donor03"),
         "runs/alignair/S1/genotype/donors/reads.donor03.fasta"),
        (layout.donor_predictions("alignair", "S1", "genotype", "donor03"),
         "runs/alignair/S1/genotype/donors/predictions.donor03.tsv"),
        (layout.donor_logs_dir("alignair", "S1", "genotype", "donor03"), "runs/alignair/S1/genotype/logs/donor03"),
        (layout.scores_dir("S1", "genotype"), "scores/S1/genotype"),
        (layout.tool_scores("S1", "shared", "igblast"), "scores/S1/shared/igblast.scores.jsonl.gz"),
        (layout.tool_report_json("S1", "shared", "igblast"), "scores/S1/shared/igblast.report.json"),
        (layout.tool_report_md("S1", "shared", "igblast"), "scores/S1/shared/igblast.report.md"),
        (layout.comparison_json("S1", "genotype"), "scores/S1/genotype/comparison.json"),
        (layout.comparison_md("S1", "genotype"), "scores/S1/genotype/comparison.md"),
        (layout.speed_dir, "speed"),
        (layout.speed_json, "speed/speed.json"),
        (layout.speed_reads(4000), "speed/reads_4000.fasta"),
        (layout.speed_work_dir("alignair-local_cuda"), "speed/runs/alignair-local_cuda"),
        (layout.report_json, "report.json"),
        (layout.report_md, "report.md"),
        (layout.diagnostics_root, "diagnostics"),
        (layout.diagnostics_dir("0a1b2c3d4e5f-6a7b8c9d0e1f"), "diagnostics/0a1b2c3d4e5f-6a7b8c9d0e1f"),
        (layout.diagnostics_suite_dir("k", "S1"), "diagnostics/k/S1"),
        (layout.diagnostics_suite_json("k", "S1"), "diagnostics/k/S1/suite.json"),
        (layout.diagnostics_suite_cases("k", "S1"), "diagnostics/k/S1/cases.jsonl.gz"),
        (layout.diagnostics_against_md("k", "S1", "o"), "diagnostics/k/S1/against_o.md"),
        (layout.diagnostics_profile_json("k"), "diagnostics/k/profile.json"),
        (layout.diagnostics_json("k"), "diagnostics/k/diagnostics.json"),
        (layout.diagnostics_md("k"), "diagnostics/k/diagnostics.md"),
        (layout.diagnostics_suite_md("k", "S1"), "diagnostics/k/S1/report.md"),
        (layout.diagnostics_profile_md("k"), "diagnostics/k/profile.md"),
        (layout.real_root, "real"),
        (layout.real_donor_dir("SC-23"), "real/SC-23"),
        (layout.real_genotype_json("W-97"), "real/W-97/genotype.json"),
        (layout.real_reads("SC-23"), "real/SC-23/reads.fasta"),
        (layout.real_reads_meta("SC-23"), "real/SC-23/reads.tsv"),
        (layout.real_preprocess_json("SC-23"), "real/SC-23/preprocess.json"),
        (layout.real_preprocess_logs("SC-23"), "real/SC-23/preprocess_logs"),
        (layout.real_preprocess_work("SC-23"), "real/SC-23/preprocess_work"),
        (layout.real_manifest(), "real/manifest.json"),
        (layout.real_summary_json(), "real/real_summary.json"),
        (layout.real_report_md(), "real/real_report.md"),
        (layout.real_setup_dir(), "real/_setup"),
        (layout.real_fetch_json("W-97"), "real/W-97/fetch.json"),
        (layout.real_truth("W-97"), "real/W-97/truth.jsonl.gz"),
        (layout.real_run_dir("igblast", "W-97", "shared"), "real/W-97/runs/igblast/shared"),
        (layout.real_predictions("alignair-api", "W-97", "genotype"),
         "real/W-97/runs/alignair-api/genotype/predictions.tsv"),
        (layout.real_run_json("alignair", "SC-23", "shared"), "real/SC-23/runs/alignair/shared/run.json"),
        (layout.real_logs_dir("igblast", "SC-23", "genotype"), "real/SC-23/runs/igblast/genotype/logs"),
        (layout.real_scores("W-97", "shared", "igblast"), "real/W-97/scores/shared/igblast.scores.jsonl.gz"),
        (layout.real_comparison_json("W-97", "shared", "igblast_vs_alignair"),
         "real/W-97/scores/shared/igblast_vs_alignair.json"),
    ]
    for path, relative in expected:
        assert path == tmp_path / relative, relative


def test_layout_is_a_pure_path_schema(tmp_path):
    root = tmp_path / "not_created"
    layout = RunLayout(str(root))
    assert layout.root == root and isinstance(layout.root, Path)
    assert layout.cases("S0") == root / "suites" / "S0" / "cases.jsonl.gz"
    assert not root.exists()


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../S1"])
def test_layout_rejects_names_that_are_not_one_component(tmp_path, name):
    layout = RunLayout(tmp_path)
    with pytest.raises(ValueError, match="path component"):
        layout.suite_dir(name)
    with pytest.raises(ValueError, match="path component"):
        layout.run_dir(name, "S1", "shared")
    with pytest.raises(ValueError, match="path component"):
        layout.speed_work_dir(name)
    with pytest.raises(ValueError, match="path component"):
        layout.work_dir(name)
    with pytest.raises(ValueError, match="path component"):
        layout.donor_reads("alignair", "S1", "genotype", name)
    with pytest.raises(ValueError, match="path component"):
        layout.tool_scores("S1", "shared", name)
    with pytest.raises(ValueError, match="path component"):
        layout.diagnostics_dir(name)
    with pytest.raises(ValueError, match="path component"):
        layout.diagnostics_suite_json("k", name)
    with pytest.raises(ValueError, match="path component"):
        layout.diagnostics_suite_cases("k", name)
    with pytest.raises(ValueError, match="path component"):
        layout.diagnostics_against_md("k", "S1", name)
    with pytest.raises(ValueError, match="path component"):
        layout.real_donor_dir(name)
    with pytest.raises(ValueError, match="path component"):
        layout.real_genotype_json(name)
    with pytest.raises(ValueError, match="path component"):
        layout.real_reads(name)
    with pytest.raises(ValueError, match="path component"):
        layout.real_preprocess_work(name)
    for method in (layout.real_fetch_json, layout.real_truth):
        with pytest.raises(ValueError, match="path component"):
            method(name)
    for args in ((name, "W-97", "shared"), ("igblast", name, "shared"), ("igblast", "W-97", name)):
        with pytest.raises(ValueError, match="path component"):
            layout.real_run_json(*args)
    for args in ((name, "shared", "igblast"), ("W-97", name, "igblast"), ("W-97", "shared", name)):
        with pytest.raises(ValueError, match="path component"):
            layout.real_scores(*args)
        with pytest.raises(ValueError, match="path component"):
            layout.real_comparison_json(*args)
