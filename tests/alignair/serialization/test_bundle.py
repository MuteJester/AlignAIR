import json
import torch
import pytest
from alignair.config.model_config import ModelConfig
from alignair.serialization.bundle import (
    save_bundle, load_bundle, compute_fingerprint, TrainingMeta, BUNDLE_FORMAT_VERSION,
)


def _cfg():
    return ModelConfig(max_seq_length=16, v_allele_count=5, j_allele_count=3,
                       d_allele_count=4, has_d_gene=True)


def test_save_creates_expected_files(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(3)},
                dataconfig={"ref": "x"}, training_meta=TrainingMeta(epochs_trained=2))
    for name in ("model.pt", "model_config.json", "dataconfig.pkl",
                 "training_meta.json", "VERSION", "fingerprint.txt"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "VERSION").read_text().strip() == str(BUNDLE_FORMAT_VERSION)


def test_roundtrip_config_dataconfig_meta(tmp_path):
    cfg = _cfg()
    save_bundle(tmp_path, model_config=cfg, state_dict={"w": torch.ones(2)},
                dataconfig={"ref": "abc"}, training_meta=TrainingMeta(epochs_trained=5, final_loss=1.5))
    loaded_cfg, dataconfig, meta = load_bundle(tmp_path)
    assert loaded_cfg == cfg
    assert dataconfig == {"ref": "abc"}
    assert meta.epochs_trained == 5 and meta.final_loss == 1.5


def test_no_dataconfig_loads_none(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(1)})
    _, dataconfig, _ = load_bundle(tmp_path)
    assert dataconfig is None


def test_fingerprint_detects_tampering(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(1)})
    # Corrupt the config after fingerprinting.
    (tmp_path / "model_config.json").write_text(json.dumps({"max_seq_length": 999}))
    with pytest.raises(ValueError):
        load_bundle(tmp_path)


def test_fingerprint_stable_for_unchanged_bundle(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(1)})
    stored = (tmp_path / "fingerprint.txt").read_text().strip()
    assert compute_fingerprint(tmp_path) == stored
