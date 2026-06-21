import json

import pytest

torch = pytest.importorskip("torch")

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.serialization.dnalignair_bundle import (
    save_dnalignair_bundle, load_dnalignair_bundle, is_bundle)


def _model():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64, aligner="softdp")
    return DNAlignAIR(cfg)


def test_bundle_roundtrip(tmp_path):
    model = _model()
    cal = {"V": {"temperature": 1.0, "epsilon": 1.1}}
    d = tmp_path / "bundle"
    save_dnalignair_bundle(d, model=model, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH",
                           calibration=cal, notes="test")
    assert is_bundle(str(d))
    b = load_dnalignair_bundle(str(d), build=True, device="cpu")
    assert b["dataconfigs"] == ["HUMAN_IGH_OGRDB"] and b["locus"] == "IGH"
    assert b["calibration"] == cal and b["meta"]["notes"] == "test"
    # config round-trips and weights load identically
    assert b["config"].to_dict() == model.config.to_dict()
    for k, v in model.state_dict().items():
        assert torch.allclose(b["model"].state_dict()[k], v)


def test_bundle_fingerprint_detects_tamper(tmp_path):
    d = tmp_path / "bundle"
    save_dnalignair_bundle(d, model=_model(), dataconfigs=["HUMAN_IGH_OGRDB"])
    # corrupt a bundle file after fingerprinting
    (d / "config.json").write_text(json.dumps({"d_model": 999}))
    with pytest.raises(ValueError, match="fingerprint"):
        load_dnalignair_bundle(str(d))


def test_bundle_missing_file_raises(tmp_path):
    d = tmp_path / "bundle"
    save_dnalignair_bundle(d, model=_model(), dataconfigs=["HUMAN_IGH_OGRDB"])
    (d / "model.pt").unlink()
    with pytest.raises(FileNotFoundError):
        load_dnalignair_bundle(str(d))


def test_is_bundle_false_for_raw_ckpt(tmp_path):
    ck = tmp_path / "m.pt"
    torch.save({"model": _model().state_dict(), "config": _model().config.to_dict()}, ck)
    assert not is_bundle(str(ck))
