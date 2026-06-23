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


def test_factored_curriculum_drives_training_and_advances():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    fc = FactoredCurriculum(start_pace=0.2)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0, curriculum=fc)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8)
    history = trainer.fit(total_steps=10)          # runs with the factored curriculum
    assert len(history) == 10
    # a high competence field advances the SHM axis (its mapped cell clears threshold)
    moved = trainer.advance_curriculum({"heavy_shm_fulllen": {"S": 0.9}, "clean": {"S": 0.9},
                                        "fragment": {"S": 0.9}}, threshold=0.7, step=0.1)
    assert "mutation_count" in moved
    assert fc.pace["mutation_count"] > 0.2
