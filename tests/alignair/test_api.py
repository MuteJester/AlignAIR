import csv

import pytest

torch = pytest.importorskip("torch")
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata

import alignair
from alignair import load_model, predict, ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle


def _tiny():
    torch.manual_seed(0)
    return DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))


def _bundle(tmp_path, locus="IGH"):
    d = tmp_path / "bundle"
    save_dnalignair_bundle(d, model=_tiny(), dataconfigs=["HUMAN_IGH_OGRDB"], locus=locus)
    return str(d)


def test_public_exports():
    for name in ("load_model", "predict", "ReferenceSet", "read_sequences", "write_airr"):
        assert hasattr(alignair, name)


def test_load_model_and_predict(tmp_path):
    loaded = load_model(_bundle(tmp_path), device="cpu")
    assert loaded.locus == "IGH" and loaded.reference_set is not None
    res = predict(loaded, ["ACGT" * 40, "TTGCAACGTACG" * 8])
    assert len(res) == 2 and len(res.sequences) == 2 and res.locus == "IGH"
    assert res.predictions[0]["v_call"] in set(loaded.reference_set.gene("V").names)
    out = tmp_path / "out.tsv"
    res.to_airr(str(out), ["r1", "r2"])
    assert out.exists() and out.read_text().startswith("sequence_id\t")


def test_predict_requires_a_reference(tmp_path):
    ck = tmp_path / "m.pt"
    m = _tiny()
    torch.save({"model": m.state_dict(), "config": m.config.to_dict()}, ck)
    loaded = load_model(str(ck), device="cpu")          # raw checkpoint: no embedded reference
    assert loaded.reference_set is None
    with pytest.raises(ValueError, match="no reference"):
        predict(loaded, ["ACGT" * 40])
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    assert len(predict(loaded, ["ACGT" * 40], reference=rs)) == 1


def test_cli_predict_matches_python_api(tmp_path):
    from alignair import cli
    bundle = _bundle(tmp_path)
    reads = tmp_path / "reads.fasta"
    reads.write_text(">r1\n" + "ACGTACGT" * 20 + "\n>r2\n" + "TTGCAACGTACG" * 10 + "\n")
    out = tmp_path / "cli.tsv"
    cli.main(["predict", str(reads), "-o", str(out), "--model", bundle, "--device", "cpu", "--quiet"])
    cli_calls = [r["v_call"] for r in csv.DictReader(out.open(), delimiter="\t")]
    loaded = load_model(bundle, device="cpu")
    from alignair import read_sequences
    _, seqs, _ = read_sequences(str(reads))
    api_calls = [p["v_call"] for p in predict(loaded, seqs).predictions]
    assert cli_calls == api_calls                        # CLI is a client of the same API


def test_cli_locus_mismatch_is_error(tmp_path):
    from alignair import cli
    bundle = _bundle(tmp_path, locus="IGH")
    igk = tmp_path / "igk.yaml"                           # an IGK reference vs an IGH model
    igk.write_text("v:\n  IGKV1-5*01: ACGTACGTACGTACGT\nj:\n  IGKJ1*01: TTTTGGGG\n")
    reads = tmp_path / "reads.fasta"
    reads.write_text(">r1\n" + "ACGTACGT" * 20 + "\n")
    out = tmp_path / "o.tsv"
    with pytest.raises(SystemExit, match="locus"):
        cli.main(["predict", str(reads), "-o", str(out), "--model", bundle,
                  "--genotype", str(igk), "--device", "cpu", "--quiet"])
    # --force-locus-mismatch proceeds
    cli.main(["predict", str(reads), "-o", str(out), "--model", bundle, "--genotype", str(igk),
              "--device", "cpu", "--quiet", "--force-locus-mismatch"])
    assert out.exists()
