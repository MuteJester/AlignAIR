import pytest
import torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.factored import FactoredCurriculum
from alignair.training.gym_trainer import GymTrainer


def test_advance_triggers_sigma_freeze():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128))
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    fc = FactoredCurriculum(start_pace=0.2)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0, curriculum=fc)
    trainer = GymTrainer(model, loss_fn, rs, gym, batch_size=8, sigma_freeze_steps=5)
    moved = trainer.advance_curriculum({"heavy_shm_fulllen": {"S": 0.9}, "clean": {"S": 0.9}})
    assert moved and trainer._freeze_remaining == 5
    assert all(w._frozen for w in loss_fn.weights.values())   # frozen now
    trainer.fit(total_steps=6, progress=False)                # 5 frozen steps then unfreeze
    assert trainer._freeze_remaining == 0
    assert not any(w._frozen for w in loss_fn.weights.values())
