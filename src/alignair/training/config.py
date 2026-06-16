"""Training configuration and reproducibility helper."""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass(eq=True)
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.0
    use_amp: bool = False
    grad_clip_norm: float = 10.0
    steps_per_epoch: int | None = None
    checkpoint_dir: str | None = None
    early_stopping_patience: int | None = None
    log_every: int = 10
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**d)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
