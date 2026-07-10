"""Phase 1 / Task 8: predict resolves --model id|path + writes run provenance."""
import hashlib
import json
import os

import GenAIRR.data as gd
import pytest

from alignair import model_file as mf
from alignair.cli.main import build_parser
from alignair.core import AlignAIR
from alignair.core.config import AlignAIRConfig
from alignair.reference.reference_set import ReferenceSet


def _run(argv):
    args = build_parser().parse_args(argv)
    return args.func(args)


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    tp = tmp_path_factory.mktemp("reg")
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    d = tp / "registry"
    (d / "human-igh").mkdir(parents=True)
    art = d / "human-igh" / "1.0.0.alignair"
    mf.save_model(str(art), AlignAIR(cfg).eval(), dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 1, "batch_size": 1}, include_trusted_pickle=False,
                  model_id="human-igh", model_version="1.0.0", card={"species": "homo_sapiens", "locus": "IGH"})
    sha = hashlib.sha256(art.read_bytes()).hexdigest()
    (d / "registry.json").write_text(json.dumps({"models": {"human-igh": {"latest": "1.0.0", "versions": {
        "1.0.0": {"file": "human-igh/1.0.0.alignair", "artifact_sha256": sha}}}}}))
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    fasta = tp / "reads.fasta"
    fasta.write_text(f">r1\n{ref.gene('V').sequences[0]}GGGGGG{ref.gene('J').sequences[0]}\n")
    return f"file://{d}", str(fasta), str(art)


def test_predict_by_registry_id_writes_airr_and_provenance(reg, tmp_path, monkeypatch):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    src, fasta, _ = reg
    out = str(tmp_path / "out.tsv")
    assert _run(["predict", "--model", "human-igh@1.0.0", "--registry", src,
                 "--input", fasta, "--out", out, "--device", "cpu"]) == 0
    assert os.path.exists(out)
    prov = json.load(open(out + ".run.json"))
    assert prov["model_id"] == "human-igh" and prov["model_version"] == "1.0.0"
    assert len(prov["artifact_sha256"]) == 64 and prov["allele_order_sha256"] and prov["n_reads"] == 1
    assert prov["command"]["input"] == fasta


def test_predict_by_path_and_no_run_metadata(reg, tmp_path):
    _, fasta, art = reg
    out = str(tmp_path / "out2.tsv")
    assert _run(["predict", "--model", art, "--input", fasta, "--out", out,
                 "--device", "cpu", "--no-run-metadata"]) == 0
    assert os.path.exists(out) and not os.path.exists(out + ".run.json")


def test_predict_offline_uncached_id_errors(reg, tmp_path, monkeypatch):
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache_empty"))
    _, fasta, _ = reg
    out = str(tmp_path / "o.tsv")
    assert _run(["predict", "--model", "human-igh@1.0.0", "--registry", "hf://unreachable/x",
                 "--input", fasta, "--out", out, "--offline", "--device", "cpu"]) == 1
    assert not os.path.exists(out)
