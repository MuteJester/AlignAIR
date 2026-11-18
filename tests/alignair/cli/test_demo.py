"""``alignair demo`` trains through ``run_training``, which seeds the run: the same command trains the same model."""
import pytest
import torch

from alignair.cli.main import main
from alignair.model_file import load_model

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _demo_weights(out):
    assert main(["demo", "--out", str(out), "--steps", "1", "--reads", "2", "--device", "cpu"]) == 0
    return load_model(str(out / "bundle" / "model.alignair"), device="cpu").model.state_dict()


def test_two_demo_runs_train_the_same_model(tmp_path):
    first, second = _demo_weights(tmp_path / "a"), _demo_weights(tmp_path / "b")
    assert first.keys() == second.keys()
    assert [key for key in first if not torch.equal(first[key], second[key])] == []    # names the weights that differ
