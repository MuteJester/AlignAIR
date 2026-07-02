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


def test_seed_extend_model_trains_a_few_steps():
    # narrow pre-retrain sanity: the refactored seed_extend stack (shared encoder, band head,
    # banded DP, band-offset loss) trains end-to-end and the loss decreases.
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)   # V/J only -> smaller/faster
    model = DNAlignAIR(cfg)
    assert getattr(model, "germline_encoder", None) is None     # no double-encode
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8)
    history = trainer.fit(total_steps=24)
    assert len(history) == 24
    assert "band" in history[-1]                                # band-offset loss is wired
    first = sum(h["total"] for h in history[:6]) / 6
    last = sum(h["total"] for h in history[-6:]) / 6
    assert last < first, f"seed_extend loss did not decrease: {first} -> {last}"


def test_seed_extend_reader_training_runs():
    # the seed_extend reader path (banded alignment_score set-NCE) trains end-to-end
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8, reader=True)
    history = trainer.fit(total_steps=20)
    assert "reader" in history[-1] and "band" in history[-1]    # both seed_extend losses wired
    first = sum(h["total"] for h in history[:5]) / 5
    last = sum(h["total"] for h in history[-5:]) / 5
    assert last < first
