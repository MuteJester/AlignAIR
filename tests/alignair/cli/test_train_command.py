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
