"""The example gallery (#106): each new example actually runs."""
import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("GenAIRR")

from alignair import cli
from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "examples"


@pytest.fixture(scope="module")
def igh_bundle(tmp_path_factory):
    torch.manual_seed(0)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    d = tmp_path_factory.mktemp("bundle")
    save_dnalignair_bundle(d, model=model, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH")
    return str(d)


def test_novel_allele_genotype_predicts(igh_bundle, tmp_path):
    geno = EX / "novel_allele" / "donor_with_novel.yaml"
    import yaml
    assert any("NOVEL" in k for k in yaml.safe_load(geno.read_text())["v"])   # the novel allele is present
    out = tmp_path / "out.tsv"
    cli.main(["predict", str(EX / "reads.fasta"), "-o", str(out), "--model", igh_bundle,
              "--genotype", str(geno), "--device", "cpu", "--quiet"])
    rows = list(csv.DictReader(out.open(), delimiter="\t"))
    assert rows and all(r["v_call"] for r in rows)


def test_custom_reference_fasta_builds_via_plan(capsys):
    code = 0
    try:
        cli.main(["train", "--v-fasta", str(EX / "custom_reference" / "v.fasta"),
                  "--j-fasta", str(EX / "custom_reference" / "j.fasta"),
                  "--d-fasta", str(EX / "custom_reference" / "d.fasta"),
                  "--chain-type", "BCR_HEAVY", "-o", "/tmp/cr_plan", "--preset", "smoke",
                  "--plan", "--device", "cpu", "--allow-curatable"])
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    out = capsys.readouterr().out
    assert code == 0
    assert "custom FASTA (chain_type=BCR_HEAVY)" in out      # the reference was built from FASTA


def test_compare_example_reports_set_rescue(capsys):
    cli.main(["compare", "--a", str(EX / "compare" / "alignair.tsv"),
              "--b", str(EX / "compare" / "igblast.tsv"), "--a-name", "AlignAIR", "--b-name", "IgBLAST"])
    report = capsys.readouterr().out
    assert "set-rescue" in report and "AlignAIR vs IgBLAST" in report
    assert "r2" in report                                    # the rescued V disagreement


def test_gallery_index_links_new_examples():
    readme = (EX / "README.md").read_text()
    for d in ("novel_allele", "custom_reference", "compare"):
        assert d in readme and (EX / d / "README.md").exists()
