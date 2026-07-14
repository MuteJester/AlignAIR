"""AIRR-review #6: custom-reference training support — build a DataConfig from V/D/J FASTAs, the
pre-training plan, and the pickle-free bundle export."""
import os

import pytest

from alignair.train.build import build_dataconfigs, training_plan

_V = "examples/custom_reference/v.fasta"
_D = "examples/custom_reference/d.fasta"
_J = "examples/custom_reference/j.fasta"
_have_fasta = all(os.path.exists(p) for p in (_V, _D, _J))


def test_build_dataconfigs_from_builtin_names():
    dcs, report = build_dataconfigs(dataconfig=["HUMAN_IGH_OGRDB"])
    assert len(dcs) == 1 and report is None and dcs[0].metadata.has_d


def test_build_dataconfigs_unknown_name_errors():
    with pytest.raises(ValueError, match="unknown built-in dataconfig"):
        build_dataconfigs(dataconfig=["NOT_A_REAL_CONFIG"])


def test_build_dataconfigs_custom_needs_companions():
    with pytest.raises(ValueError, match="needs --v-fasta"):
        build_dataconfigs(v_fasta=_V)                 # missing --j-fasta / --chain-type


@pytest.mark.skipif(not _have_fasta, reason="custom_reference example FASTAs not present")
def test_build_dataconfigs_from_custom_fasta():
    dcs, report = build_dataconfigs(v_fasta=_V, j_fasta=_J, d_fasta=_D, chain_type="BCR_HEAVY")
    assert len(dcs) == 1 and dcs[0].metadata.has_d
    assert report is not None                          # cartridge build report (warnings/rejected)


def test_training_plan_reports_reference_and_params():
    dcs, _ = build_dataconfigs(dataconfig=["HUMAN_IGH_OGRDB"])
    plan = training_plan(dcs, steps=123, batch_size=8)
    assert plan["loci"] == ["IGH"] and plan["has_d"] is True
    assert plan["alleles"]["V"] > 0 and plan["model_parameters"] > 0
    assert plan["steps"] == 123 and plan["batch_size"] == 8


def test_train_plan_cli_does_not_train(tmp_path):
    from alignair.cli.main import main
    out = tmp_path / "r"
    rc = main(["train", "--dataconfig", "HUMAN_IGH_OGRDB", "--out", str(out), "--preset", "quick",
               "--plan", "--device", "cpu"])
    assert rc == 0
    assert not (out / "model.alignair").exists()      # --plan validates only, never trains


@pytest.mark.slow
@pytest.mark.skipif(not _have_fasta, reason="custom_reference example FASTAs not present")
def test_custom_fasta_train_and_pickle_free_export(tmp_path):
    from alignair import model_file as mf
    from alignair.api import load_model
    from alignair.cli.main import main
    out = tmp_path / "run"
    rc = main(["train", "--v-fasta", _V, "--j-fasta", _J, "--d-fasta", _D, "--chain-type", "BCR_HEAVY",
               "--out", str(out), "--preset", "quick", "--steps", "4", "--batch-size", "4",
               "--val-every", "2", "--device", "cpu"])
    assert rc == 0
    bundle = out / "bundle"
    for f in ("model.alignair", "model_card.md", "reference_manifest.json", "validation_report.json"):
        assert (bundle / f).exists(), f
    secs = mf.read_metadata(str(bundle / "model.alignair")).get("sections", {})
    assert not any(k.startswith("dataconfig/") or k == "train_state" for k in secs)   # pickle-free
    model, ref = load_model(str(bundle / "model.alignair"), device="cpu")             # loads w/o trust
    assert len(ref.gene("V")) == 8
