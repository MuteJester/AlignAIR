import logging
import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.training.gym_trainer import GymTrainer


def test_train_few_steps_loss_decreases(caplog):
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)  # V/J only -> smaller/faster
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8)
    with caplog.at_level(logging.INFO):
        history = trainer.fit(total_steps=20)
    assert len(history) == 20
    first = sum(h["total"] for h in history[:5]) / 5
    last = sum(h["total"] for h in history[-5:]) / 5
    assert last < first, f"loss did not decrease: {first} -> {last}"
    assert any("curriculum stage" in m.lower() for m in caplog.messages)


def test_evaluate_reports_metrics():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=3)
    trainer = GymTrainer(model, loss_fn, rs, gym, batch_size=8)
    metrics = trainer.evaluate(n_batches=2)
    for k in ("region_acc", "v_call_agreement", "loss"):
        assert k in metrics
    assert 0.0 <= metrics["region_acc"] <= 1.0


def test_train_with_cached_reference_refresh():
    # refresh_reference_every > 1 must not hit a freed-graph error and must still learn
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8,
                         refresh_reference_every=5)
    history = trainer.fit(total_steps=20)
    assert len(history) == 20
    first = sum(h["total"] for h in history[:5]) / 5
    last = sum(h["total"] for h in history[-5:]) / 5
    assert last < first
