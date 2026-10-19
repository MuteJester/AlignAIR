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
        (layout.speed_dir, "speed"),
        (layout.speed_json, "speed/speed.json"),
        (layout.report_json, "report.json"),
        (layout.report_md, "report.md"),
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
        layout.work_dir(name)
    with pytest.raises(ValueError, match="path component"):
        layout.donor_reads("alignair", "S1", "genotype", name)
