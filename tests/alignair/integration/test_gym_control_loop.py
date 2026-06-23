"""End-to-end: the competence-gated GymController, wired to a live GymTrainer,
climbs the ladder during fit() and renders the 8-bit HUD. Lenient gates so an
under-trained smoke model still promotes — this verifies the WIRING, not accuracy."""
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
from alignair.gym.control import (
    GymConfig, GateSpec, RankLadder, PromotionGate, GymController, GymHUD)


def test_controller_climbs_and_renders_hud_during_fit():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)   # V/J only -> fast
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8)

    # trivially-passable gate so every exam promotes -> exercises the climb mechanism
    gates = (GateSpec("region_acc", "higher", (0.0, 0.0, 0.0)),)
    gconf = GymConfig(n_levels=3, gates=gates, patience=2, exam_every=5, exam_batches=1)
    ladder = RankLadder(n_levels=3)
    captured = []
    ctrl = GymController(
        gconf, ladder, PromotionGate(gates),
        evaluator=lambda level, batches: trainer.evaluate(n_batches=batches,
                                                          p=ladder.progress(level)),
        hud=GymHUD(color=False), emit=captured.append)

    trainer.fit(total_steps=20, controller=ctrl)

    assert ctrl.best_level == 2          # climbed all the way to the top floor
    assert ctrl.done is True             # top floor cleared -> gym complete
    text = "\n".join(captured)
    assert "FLOOR" in text               # HUD rendered
    assert "ROOM CLEARED" in text        # promotion callout fired
