import torch
from alignair.training.config import TrainingConfig, seed_everything


def test_defaults():
    cfg = TrainingConfig()
    assert cfg.epochs >= 1 and cfg.lr > 0
    assert cfg.use_amp in (True, False)


def test_roundtrip():
    cfg = TrainingConfig(epochs=3, lr=1e-3, batch_size=8)
    assert TrainingConfig.from_dict(cfg.to_dict()) == cfg


def test_seed_reproducible():
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.allclose(a, b)
