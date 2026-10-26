import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignair.aa.guard.checkpoint import CheckpointProvenance
from alignair.aa.guard.features import FEATURE_NAMES
from alignair.aa.guard.model import GuardModel
from alignair.aa.guard.training import engine
from alignair.errors import ArtifactIntegrityError, ConfigurationError

pytestmark = pytest.mark.component


@pytest.fixture
def run_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "GuardModel", lambda cid, **kw: GuardModel(cid, width=16, depth=1, heads=2, **kw))

    def provenance(**kwargs):
        return CheckpointProvenance(tokenizer="synthetic", feature_order=FEATURE_NAMES,
                                    class_map_hash="classes", software_versions={"torch": torch.__version__}, **kwargs)

    monkeypatch.setattr(engine, "build_provenance", provenance)
    sequences = ["QVQLVQSGAEVKKPG" + aa * 8 for aa in "ACDEFGHIKL"]
    records = [{"id": str(i), "aa": seq, "group": "positive_real" if i % 2 else "igk_igl",
                "category": "positive" if i % 2 else "igk", "expected_state": "supported" if i % 2 else "unknown_ood"}
               for i, seq in enumerate(sequences)]
    train, dev = records[:6], records[6:]
    oe = [{"id": "oe", "aa": "ACDEFGHIKLMNPQRSTVWY", "group": "oe"}]
    cache = {hashlib.sha256(r["aa"].encode()).hexdigest(): (np.arange(len(FEATURE_NAMES)) + i).tolist()
             for i, r in enumerate(records + oe)}
    manifest = {
        "training": {"max_steps": 4, "batch_size": 4, "eval_every": 2, "early_stop_patience": 10,
                     "min_delta": 0, "grad_clip": 1, "amp": False},
        "optimizer": {"lr": 0.001, "weight_decay": 0.01}, "schedule": {"warmup_steps": 1},
        "loss_weights": {"complete": 1, "app": 1, "oe": 0.5, "contrast": 0.5, "source": 0.5},
        "energy_margins": {"m_in": 5, "m_out": 10}, "grl_lambda": 1, "oe": {"cap_fraction": 0.25},
        "bank_checksum": "bank", "manifest_sha256": "manifest", "train_partition_hash": "partition",
    }
    config = engine.RunConfig(manifest, tmp_path / "checkpoints", tmp_path / "budget.json")
    return dict(cid="C4", seed=7, train=train, dev=dev, oe=oe, feat_cache=cache,
                cat2id={"igk": 0, "positive": 1}, eligible_dt=set(), n_sources=2,
                config=config, bank=SimpleNamespace(checksum=lambda: "bank"))


def test_cpu_training_writes_reusable_checkpoint_and_accounting(run_inputs):
    metrics = engine.train_one(**run_inputs)
    assert metrics["status"] == "ok"
    assert metrics["gpu_h"] == 0
    assert metrics["dev_total_loss"] < float("inf")
    config = run_inputs["config"]
    saved = torch.load(config.checkpoint_dir / "C4.s7.final.pt", weights_only=False)
    assert saved["training_contract"]
    assert saved["provenance"]["bank_checksum"] == "bank"
    assert "C4.s7" in json.loads(config.budget_path.read_text())["runs"]
    config.budget_path.unlink()
    assert engine.train_one(**run_inputs) == metrics
    assert list(json.loads(config.budget_path.read_text())["runs"]) == ["C4.s7"]


def test_interrupted_resume_matches_uninterrupted_weights_and_metrics(run_inputs, tmp_path, monkeypatch):
    config = run_inputs["config"]
    reference = engine.RunConfig(config.manifest, tmp_path / "reference", tmp_path / "reference-budget.json")
    expected = engine.train_one(**{**run_inputs, "config": reference})
    save = torch.save

    def interrupt_after_checkpoint(value, path, *args, **kwargs):
        save(value, path, *args, **kwargs)
        if path.name == "C4.s7.pt":
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupt_after_checkpoint)
        with pytest.raises(KeyboardInterrupt):
            engine.train_one(**run_inputs)
    actual = engine.train_one(**run_inputs)
    ref_state = torch.load(reference.checkpoint_dir / "C4.s7.final.pt", weights_only=False)["model"]
    state = torch.load(config.checkpoint_dir / "C4.s7.final.pt", weights_only=False)["model"]
    assert all(torch.equal(state[key], ref_state[key]) for key in state)
    for key in ("dev_total_loss", "train_loss_ema", "component_losses", "applicability_auroc"):
        assert actual[key] == expected[key]


def test_completed_checkpoint_rejects_changed_training_settings(run_inputs):
    engine.train_one(**run_inputs)
    config = run_inputs["config"]
    manifest = copy.deepcopy(config.manifest)
    manifest["loss_weights"]["app"] = 2
    changed = engine.RunConfig(manifest, config.checkpoint_dir, config.budget_path)
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        engine.train_one(**{**run_inputs, "config": changed})


def test_legacy_unbound_checkpoint_is_not_silently_reused(run_inputs):
    config = run_inputs["config"]
    config.checkpoint_dir.mkdir()
    path = config.checkpoint_dir / "C4.s7.final.pt"
    torch.save({"metrics": {"status": "ok"}}, path)
    before = path.read_bytes()
    with pytest.raises(ArtifactIntegrityError):
        engine.train_one(**run_inputs)
    assert path.read_bytes() == before


def test_nonfinite_loss_creates_failure_diagnostic_without_final_model(run_inputs, monkeypatch):
    def nonfinite(*args):
        return torch.tensor(float("nan")), {}, {}

    monkeypatch.setattr(engine, "_forward_losses", nonfinite)
    result = engine.train_one(**run_inputs)
    assert result["status"] == "engineering_failure"
    assert result["error"]["code"] == "ALIGNAIR_NUMERICAL_FAILURE"
    directory = run_inputs["config"].checkpoint_dir
    assert (directory / "C4.s7.failure.json").is_file()
    assert not (directory / "C4.s7.final.pt").exists()


def test_device_errors_are_configuration_failures(monkeypatch):
    with pytest.raises(ConfigurationError):
        engine.resolve_device("invalid-device")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert engine.resolve_device("auto") == "cpu"
    with pytest.raises(ConfigurationError, match="unavailable"):
        engine.resolve_device("cuda")
