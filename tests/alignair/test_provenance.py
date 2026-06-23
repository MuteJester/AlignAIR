"""Expanded provenance in bundle meta.json and the prediction run.json (#94)."""
import json

import pytest

from alignair import provenance as prov


def test_hash_json_stable_and_none():
    assert prov.hash_json(None) is None
    assert prov.hash_json({"a": 1, "b": 2}) == prov.hash_json({"b": 2, "a": 1})  # order-independent
    assert prov.hash_json({"a": 1}) != prov.hash_json({"a": 2})


def test_package_versions_and_git_shape():
    pv = prov.package_versions()
    assert "torch" in pv and "python" in pv
    sha = prov.git_commit_sha()
    assert sha is None or isinstance(sha, str)


def test_reference_hash_distinguishes_references():
    pytest.importorskip("GenAIRR")
    from alignair.reference.reference_set import ReferenceSet
    rs = ReferenceSet.from_genotype({"v": {"V1*01": "ACGT", "V2*01": "TTTT"},
                                     "d": {"D1*01": "GG"}, "j": {"J1*01": "CC"}})
    sub = ReferenceSet.from_genotype({"v": {"V1*01": "ACGT"}, "d": {"D1*01": "GG"},
                                      "j": {"J1*01": "CC"}})
    assert prov.reference_hash(rs) and prov.reference_hash(rs) != prov.reference_hash(sub)
    assert prov.reference_hash(None) is None


def test_bundle_meta_carries_provenance_and_training(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("GenAIRR")
    from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle, load_dnalignair_bundle
    from alignair.config.dnalignair_config import DNAlignAIRConfig
    from alignair.core.dnalignair import DNAlignAIR
    torch.manual_seed(0)
    m = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    d = tmp_path / "b"
    save_dnalignair_bundle(d, model=m, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH",
                           calibration={"V": {"temperature": 1.0}},
                           training_meta={"preset": "smoke", "seed": 7, "steps": 3})
    meta = json.loads((d / "meta.json").read_text())
    for k in ("alignair_version", "versions", "reference_hash", "config_hash",
              "calibration_hash", "created_utc", "training"):
        assert k in meta
    assert meta["training"] == {"preset": "smoke", "seed": 7, "steps": 3}
    assert meta["calibration_hash"] and meta["reference_hash"]
    # provenance additions do not break the fingerprint
    assert load_dnalignair_bundle(str(d), build=False)["meta"]["training"]["seed"] == 7


def test_run_json_provenance(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("GenAIRR")
    from alignair import cli
    from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle
    from alignair.config.dnalignair_config import DNAlignAIRConfig
    from alignair.core.dnalignair import DNAlignAIR
    torch.manual_seed(0)
    m = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    bundle = tmp_path / "bundle"
    save_dnalignair_bundle(bundle, model=m, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH")
    reads = tmp_path / "r.fasta"
    reads.write_text(">r1\n" + "ACGTACGT" * 20 + "\n")
    out = tmp_path / "o.tsv"
    cli.main(["predict", str(reads), "-o", str(out), "--model", str(bundle),
              "--device", "cpu", "--quiet"])
    run = json.loads((tmp_path / "o.tsv.run.json").read_text())
    for k in ("git_commit", "reference_hash", "calibration_hash", "cuda", "versions",
              "model_build", "model_fingerprint"):
        assert k in run
    assert run["versions"]["torch"] and run["reference_hash"]
    assert run["model_build"]["alignair_version"]            # carried from the bundle's meta.json
