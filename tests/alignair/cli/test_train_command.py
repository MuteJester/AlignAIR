"""``alignair train``: the pre-training plan, ``--plan`` that never trains, and a reference request it refuses with the
flag that sets the field."""
import pytest

from alignair.cli.main import main
from alignair.cli.train import training_plan
from alignair.genairr.custom import build_dataconfigs

pytestmark = pytest.mark.integration


def test_training_plan_reports_reference_and_params():
    dcs, _ = build_dataconfigs(dataconfig=["HUMAN_IGH_OGRDB"])
    plan = training_plan(dcs, steps=123, batch_size=8)
    assert plan["loci"] == ["IGH"] and plan["has_d"] is True
    assert plan["alleles"]["V"] > 0 and plan["model_parameters"] > 0
    assert plan["steps"] == 123 and plan["batch_size"] == 8


def test_train_plan_cli_does_not_train(tmp_path):
    out = tmp_path / "r"
    rc = main(["train", "--dataconfig", "HUMAN_IGH_OGRDB", "--out", str(out), "--preset", "quick",
               "--plan", "--device", "cpu"])
    assert rc == 0
    assert not (out / "model.alignair").exists()      # --plan validates only, never trains


def test_a_refused_reference_names_the_flag_that_sets_it(tmp_path, capsys):
    rc = main(["train", "--v-fasta", "examples/custom_reference/v.fasta", "--out", str(tmp_path / "r"), "--plan"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "needs a V FASTA, a J FASTA and a chain type" in err and "--j-fasta sets it" in err


class _Stop(Exception):
    """Raised by the stand-in run_training once it has the configuration."""


def _capture_config(monkeypatch) -> dict:
    seen = {}

    def run_training(config, **kwargs):
        seen["config"] = config
        raise _Stop
    monkeypatch.setattr("alignair.train.workflow.run_training", run_training)
    return seen


def test_train_init_from_reaches_the_training_config(tmp_path, monkeypatch):
    """``--init-from`` starts the run from a model's weights, as ``TrainingConfig(init_from=)`` does in Python."""
    from tests.alignair.predict.golden_rows import TEST_CHIP
    seen = _capture_config(monkeypatch)
    main(["train", "--dataconfig", "HUMAN_IGH_OGRDB", "--out", str(tmp_path / "run"), "--steps", "1",
          "--init-from", str(TEST_CHIP)])           # the stand-in stops the run (main reports its _Stop)
    assert str(seen["config"].init_from) == str(TEST_CHIP)


def test_train_init_from_takes_a_registry_id(tmp_path, monkeypatch):
    import hashlib
    import json
    import shutil

    from tests.alignair.predict.golden_rows import TEST_CHIP
    monkeypatch.setenv("ALIGNAIR_CACHE_DIR", str(tmp_path / "cache"))
    registry = tmp_path / "registry"
    (registry / "chip").mkdir(parents=True)
    shutil.copyfile(TEST_CHIP, registry / "chip" / "1.0.0.alignair")
    sha = hashlib.sha256(TEST_CHIP.read_bytes()).hexdigest()
    (registry / "registry.json").write_text(json.dumps({"schema": "alignair.registry.v1", "models": {"chip": {
        "latest": "1.0.0", "versions": {"1.0.0": {"file": "chip/1.0.0.alignair", "artifact_sha256": sha}}}}}))
    monkeypatch.setenv("ALIGNAIR_REGISTRY", f"file://{registry}")
    seen = _capture_config(monkeypatch)
    main(["train", "--dataconfig", "HUMAN_IGH_OGRDB", "--out", str(tmp_path / "run"), "--steps", "1",
          "--init-from", "chip@1.0.0"])
    assert hashlib.sha256(open(seen["config"].init_from, "rb").read()).hexdigest() == sha
