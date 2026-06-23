"""Workflow-wrapper drafts (#91): the shared samplesheet is a real `alignair batch` manifest,
and each wrapper is structurally valid and calls the documented CLI."""
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / "workflows"


def test_samplesheet_columns_match_batch_manifest():
    with (WF / "samplesheet.csv").open() as f:
        cols = next(csv.reader(f))
    assert cols[:2] == ["sample_id", "input"]          # the required `alignair batch` columns
    assert "genotype" in cols                          # optional per-row reference


def test_samplesheet_is_a_valid_batch_manifest(tmp_path):
    """The wrappers' input contract is the batch manifest — prove it actually runs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("GenAIRR")
    from alignair import cli
    from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle
    from alignair.config.dnalignair_config import DNAlignAIRConfig
    from alignair.core.dnalignair import DNAlignAIR
    torch.manual_seed(0)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    bundle = tmp_path / "bundle"
    save_dnalignair_bundle(bundle, model=model, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH")
    out = tmp_path / "results"
    cli.main(["batch", "--manifest", str(WF / "samplesheet.csv"), "-o", str(out),
              "--model", str(bundle), "--device", "cpu", "--quiet"])
    assert (out / "example.tsv").exists()
    rows = {r["sample_id"]: r for r in csv.DictReader((out / "manifest_summary.tsv").open(), delimiter="\t")}
    assert rows["example"]["status"] == "ok"


def test_galaxy_xml_is_well_formed_and_wraps_predict():
    tree = ET.parse(WF / "galaxy" / "alignair_predict.xml")
    root = tree.getroot()
    assert root.tag == "tool" and root.get("id") == "alignair_predict"
    cmd = root.findtext("command")
    assert "alignair predict" in cmd
    assert root.find("command").get("detect_errors") == "exit_code"   # exit codes -> job failures
    assert root.find("requirements/container") is not None            # containerised
    out = root.find("outputs/data")
    assert out.get("name") == "airr" and out.get("format") == "tabular"


def test_nextflow_draft_calls_predict():
    nf = (WF / "nextflow" / "main.nf").read_text()
    assert "process ALIGNAIR_PREDICT" in nf
    assert "alignair predict" in nf and "splitCsv(header: true)" in nf
    assert "container" in nf


def test_snakemake_draft_calls_predict():
    smk = (WF / "snakemake" / "Snakefile").read_text()
    assert "rule predict:" in smk and "rule all:" in smk
    assert "alignair predict" in smk and "container:" in smk
    if (cfg := (WF / "snakemake" / "config.yaml")).exists():
        import yaml
        c = yaml.safe_load(cfg.read_text())
        assert c["samplesheet"].endswith("samplesheet.csv") and "model" in c


def test_snakemake_dry_run_if_available(tmp_path):
    """If snakemake is installed, the Snakefile must at least parse + plan (dry run)."""
    import shutil
    import subprocess
    if not shutil.which("snakemake"):
        pytest.skip("snakemake not installed")
    r = subprocess.run(["snakemake", "-n", "--configfile", str(WF / "snakemake" / "config.yaml"),
                        "--config", "model=DUMMY", f"samplesheet={WF / 'samplesheet.csv'}"],
                       cwd=WF / "snakemake", capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
