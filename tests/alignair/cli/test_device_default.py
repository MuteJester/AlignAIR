"""``alignair predict`` (and ``demo``) without ``--device`` resolve the device as ``--device auto`` does, as the Python
API's ``device="auto"``: CUDA, then Apple MPS, then CPU. An unavailable explicit device falls back to the CPU."""
import pytest
import torch

from alignair.cli.main import build_parser
from alignair.errors import ModelError

pytestmark = pytest.mark.component


def _device_predict_loads_with(monkeypatch, tmp_path, *extra):
    from alignair.cli import predict as predict_cli
    from tests.alignair.predict.golden_rows import TEST_CHIP
    seen = {}

    def load(model_path, *, device, reference):
        seen["device"] = device
        raise ModelError("stop here")
    monkeypatch.setattr(predict_cli, "_load_reporting_locus", load)
    args = build_parser().parse_args(["predict", "--model", str(TEST_CHIP), "--input", "r.fasta",
                                      "--out", str(tmp_path / "o.tsv"), "--offline", "--quiet", *extra])
    assert args.func(args) != 0
    return seen["device"]


@pytest.fixture
def mps_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


def test_predict_without_device_picks_mps_when_it_is_the_accelerator(monkeypatch, tmp_path, mps_only):
    assert _device_predict_loads_with(monkeypatch, tmp_path) == "mps"


def test_predict_without_device_on_a_cpu_only_machine_is_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _device_predict_loads_with(monkeypatch, tmp_path) == "cpu"
    assert _device_predict_loads_with(monkeypatch, tmp_path, "--device", "cuda") == "cpu"


def test_the_help_says_auto_and_names_the_order():
    import argparse
    (sub,) = [a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)]
    for command in ("predict", "demo"):
        (device,) = [a for a in sub.choices[command]._actions if "--device" in a.option_strings]
        assert "default: auto" in device.help and "MPS" in device.help
