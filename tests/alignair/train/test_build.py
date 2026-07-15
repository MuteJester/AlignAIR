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


def test_build_dataconfigs_rejects_mixed_modes():
    with pytest.raises(ValueError, match="ONE reference mode"):
        build_dataconfigs(dataconfig=["HUMAN_IGH_OGRDB"], v_fasta="v.fa", j_fasta="j.fa",
                          chain_type="BCR_HEAVY")


def test_build_dataconfigs_d_fasta_rules():
    with pytest.raises(ValueError, match="carries a D gene"):     # heavy needs --d-fasta
        build_dataconfigs(v_fasta=_V, j_fasta=_J, chain_type="BCR_HEAVY")
    with pytest.raises(ValueError, match="no D gene"):            # light must not get a D FASTA
        build_dataconfigs(v_fasta=_V, j_fasta=_J, d_fasta=_D, chain_type="BCR_LIGHT_KAPPA")


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


def test_export_bundle_fails_closed_on_existing_dir(tmp_path):
    """A pre-existing bundle dir must not be silently destroyed: export refuses without overwrite, and
    fails BEFORE touching the checkpoint (so a bad re-run can't clobber a published bundle)."""
    from alignair.train.build import export_bundle
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.alignair").write_text("precious")
    with pytest.raises(FileExistsError, match="already exists"):
        export_bundle("/no/such/checkpoint.alignair", [], str(bundle),
                      training={"steps": 1, "batch_size": 1}, validate=False)
    assert (bundle / "model.alignair").read_text() == "precious"      # untouched (guard fired first)


def test_train_cli_refuses_existing_bundle_before_training(tmp_path):
    """`alignair train` fails closed on an existing bundle/ BEFORE spending training time (not only at
    export), so a re-run can't clobber a published bundle after a long train (AIRR-review)."""
    from alignair.cli.main import main
    out = tmp_path / "run"
    (out / "bundle").mkdir(parents=True)
    (out / "bundle" / "model.alignair").write_text("precious")
    rc = main(["train", "--dataconfig", "HUMAN_IGH_OGRDB", "--out", str(out),
               "--preset", "quick", "--device", "cpu"])
    assert rc == 1
    assert (out / "bundle" / "model.alignair").read_text() == "precious"     # untrained, untouched


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
    import json
    manifest = json.load(open(bundle / "reference_manifest.json"))                    # richer manifest
    assert manifest["allele_order_sha256"] and manifest["versions"]["alignair"]
    assert manifest["reference_fasta_sha256"]                                         # embedded-ref fingerprint
    assert manifest["model_artifact_sha256"] and "v_fasta" in manifest["sources"]
    assert manifest["genes"]["V"]["n"] == 8 and "anchored" in manifest["genes"]["V"]
