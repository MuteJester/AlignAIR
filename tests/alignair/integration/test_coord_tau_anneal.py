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


def test_coord_tau_anneals_wide_to_sharp_over_horizon():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d, coord_loss="soft")
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8)

    # EARLY in a long horizon -> tau should be WIDE (near the 3.0 start, not the fixed 1.0).
    trainer.fit(total_steps=2, global_total=200)
    assert loss_fn.coord_tau > 2.5, f"tau not wide early: {loss_fn.coord_tau}"

    # near the END of the horizon -> tau annealed SHARP (toward 0.75).
    trainer._global_step = 199
    trainer.fit(total_steps=2, global_total=200)
    assert loss_fn.coord_tau < 1.0, f"tau not sharp late: {loss_fn.coord_tau}"
