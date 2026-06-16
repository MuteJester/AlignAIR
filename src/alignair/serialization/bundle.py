"""PyTorch model-bundle IO: state_dict + config + dataconfig + meta + fingerprint."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

from ..config.model_config import ModelConfig

BUNDLE_FORMAT_VERSION = 1
_REQUIRED = ("model.pt", "model_config.json", "VERSION", "fingerprint.txt")


@dataclass
class TrainingMeta:
    epochs_trained: int = 0
    best_loss: Optional[float] = None
    final_loss: Optional[float] = None
    metrics_summary: dict = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMeta":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def compute_fingerprint(bundle_dir) -> str:
    """SHA-256 over every bundle file except fingerprint.txt, in name order."""
    h = hashlib.sha256()
    for p in sorted(Path(bundle_dir).iterdir()):
        if not p.is_file() or p.name == "fingerprint.txt":
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def save_bundle(bundle_dir, *, model_config: ModelConfig, state_dict,
                dataconfig=None, training_meta: Optional[TrainingMeta] = None) -> None:
    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, d / "model.pt")
    (d / "model_config.json").write_text(
        json.dumps(model_config.to_dict(), indent=2, sort_keys=True))
    if dataconfig is not None:
        with (d / "dataconfig.pkl").open("wb") as f:
            pickle.dump(dataconfig, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = training_meta or TrainingMeta()
    (d / "training_meta.json").write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True))
    (d / "VERSION").write_text(str(BUNDLE_FORMAT_VERSION))
    (d / "fingerprint.txt").write_text(compute_fingerprint(d))


def load_bundle(bundle_dir):
    """Return (ModelConfig, dataconfig | None, TrainingMeta); verify fingerprint."""
    d = Path(bundle_dir)
    missing = [n for n in _REQUIRED if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(f"bundle missing required files: {missing}")
    expected = (d / "fingerprint.txt").read_text().strip()
    if compute_fingerprint(d) != expected:
        raise ValueError(f"bundle fingerprint mismatch — {d} was modified or is corrupt")

    model_config = ModelConfig.from_dict(json.loads((d / "model_config.json").read_text()))
    dataconfig = None
    if (d / "dataconfig.pkl").exists():
        with (d / "dataconfig.pkl").open("rb") as f:
            dataconfig = pickle.load(f)
    meta = TrainingMeta()
    if (d / "training_meta.json").exists():
        meta = TrainingMeta.from_dict(json.loads((d / "training_meta.json").read_text()))
    return model_config, dataconfig, meta
