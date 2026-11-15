"""The pickle-free bundle export, including the provenance it must carry."""
import os

import pytest


pytestmark = pytest.mark.integration

_V = "examples/custom_reference/v.fasta"
_D = "examples/custom_reference/d.fasta"
_J = "examples/custom_reference/j.fasta"
_have_fasta = all(os.path.exists(p) for p in (_V, _D, _J))


def test_export_bundle_fails_closed_on_existing_dir(tmp_path):
    """A pre-existing bundle dir must not be silently destroyed: export refuses without overwrite, and
    fails BEFORE touching the checkpoint (so a bad re-run can't clobber a published bundle)."""
    from alignair.train.bundle import export_bundle
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.alignair").write_text("precious")
    with pytest.raises(FileExistsError, match="already exists"):
        export_bundle("/no/such/checkpoint.alignair", [], str(bundle), validate=False)
    assert (bundle / "model.alignair").read_text() == "precious"      # untouched (guard fired first)


def test_train_cli_refuses_existing_bundle_before_training(tmp_path):
    """`alignair train` fails closed on an existing bundle/ BEFORE spending training time (not only at
    export), so a re-run can't clobber a published bundle after a long train."""
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


def test_checkpoint_training_tolerates_unreadable_artifact():
    """Provenance lookup must never block an export (legacy .pt / missing file -> empty, not raise)."""
    from alignair.train.bundle import _checkpoint_training
    assert _checkpoint_training("/no/such/checkpoint.alignair") == {}


@pytest.mark.slow
def test_bundle_export_preserves_checkpoint_provenance(tmp_path):
    """The resumable checkpoint and the distributable bundle must record the SAME training provenance.

    `effective_mutation_caps` is the one that matters: a TRA run receives zero SHM, and a bundle that
    dropped that field would read as an uncapped curriculum with heavy_shm=0.25. The bundle used to be
    built from a caller-reconstructed training dict, which silently discarded `train_args` entirely.
    """
    import GenAIRR.data as gd

    from alignair import TrainingConfig, run_training
    from alignair import model_file as mf
    from alignair.train.bundle import export_bundle

    out = tmp_path / "run"
    cfg = TrainingConfig.from_genairr("HUMAN_TCRA_IMGT", preset="quick", steps=2, batch_size=2)
    run = run_training(cfg, output_dir=str(out))
    bundle = export_bundle(run.model_path, [gd.HUMAN_TCRA_IMGT], str(out / "bundle"), validate=False)

    ck = mf.read_metadata(run.model_path)["training"]
    bd = mf.read_metadata(bundle)["training"]

    assert ck["train_args"]["effective_mutation_caps"] == {"HUMAN_TCRA_IMGT": 0.0}
    assert bd["train_args"]["effective_mutation_caps"] == ck["train_args"]["effective_mutation_caps"]
    for k in ("progresses", "heavy_shm", "seed", "mutation_cap"):   # the rest survives too
        assert bd["train_args"][k] == ck["train_args"][k], f"bundle lost train_args.{k}"

    card = (out / "bundle" / "model_card.md").read_text()           # human-facing provenance
    assert "effective SHM cap per locus" in card
    assert "HUMAN_TCRA_IMGT=0.0" in card
    # a chip reads its alleles from the bound reference: the card says so, not that alleles need a retrain
    assert "alignair predict --reference" in card and "requires training a new model" not in card
