import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.training.gym_trainer import GymTrainer


def test_distillation_runs_and_teacher_tracks_student():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=48, n_layers=2, nhead=4, dim_feedforward=96)
    rs = __import__("alignair.reference.reference_set", fromlist=["ReferenceSet"]).ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8,
                         distill=True, distill_weight=1.0)
    # teacher starts identical to student
    t0 = next(trainer.teacher.model.parameters()).clone()
    history = trainer.fit(total_steps=12)
    assert len(history) == 12
    assert all("distill" in h for h in history), "distill term not recorded"
    assert all(torch.isfinite(torch.tensor(h["distill"])) for h in history)
    # teacher moved (tracked the student) and stays grad-free
    t1 = next(trainer.teacher.model.parameters())
    assert (t1 - t0).abs().sum() > 0
    assert not any(p.requires_grad for p in trainer.teacher.model.parameters())


def test_training_without_distill_has_no_distill_term():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=48, n_layers=2, nhead=4, dim_feedforward=96)
    rs = __import__("alignair.reference.reference_set", fromlist=["ReferenceSet"]).ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    trainer = GymTrainer(DNAlignAIR(cfg), DNAlignAIRLoss(has_d=rs.has_d), rs,
                         AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, seed=0), batch_size=8)
    history = trainer.fit(total_steps=6)
    assert trainer.teacher is None
    assert all("distill" not in h for h in history)
