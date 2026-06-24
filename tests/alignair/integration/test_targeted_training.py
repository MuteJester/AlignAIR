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
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.targeting import TargetedCurriculum
from alignair.training.gym_trainer import GymTrainer


def test_targeted_curriculum_trains_and_updates_targets():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128))
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    tc = TargetedCurriculum(FactoredCurriculum(start_pace=0.3), FrozenLattice.standard(seed=0))
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0, curriculum=tc)
    trainer = GymTrainer(model, loss_fn, rs, gym, batch_size=8)
    trainer.fit(total_steps=6, progress=False)               # mixture sampling works in training
    trainer.advance_curriculum({"clean": {"S": 0.97}, "heavy_shm": {"S": 0.40},
                                "heavy_shm_fulllen": {"S": 0.95}})
    assert tc.tracker.top_cell() == "heavy_shm"              # targeting updated from the field
    assert any(abs(w - tc.p_alp) < 1e-9 for w, _ in tc.components())
