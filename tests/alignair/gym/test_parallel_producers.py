"""Shared-state multiprocessing producer pool for the gym."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.factored import FactoredCurriculum
from alignair.training.gym_trainer import GymTrainer


def _rs():
    return ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)


def test_default_gym_has_no_shared_state():
    # the single-process path must NOT spawn a Manager (LatticeEvaluator makes many gyms)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], _rs(), n=8, seed=0)
    assert gym._shared_params is None


def test_enable_sharing_pushes_params_and_bumps_version():
    fc = FactoredCurriculum(start_pace=0.2)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], _rs(), n=8, seed=0, curriculum=fc, shared=True)
    assert gym._shared_params is not None
    v0 = gym._version.value
    gym.set_progress(0.5)
    assert gym._version.value == v0 + 1                  # producers signalled
    # advancing a factored axis then refreshing pushes the new floor
    fc.pace["mutation_count"] = 1.0
    gym.refresh_params()
    assert gym._version.value == v0 + 2
    assert gym._shared_params["params"]["mutation_count"] == fc.params()["mutation_count"]


def test_enable_sharing_is_idempotent():
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], _rs(), n=8, seed=0, shared=True)
    sp = gym._shared_params
    gym.enable_sharing()
    assert gym._shared_params is sp                      # no second Manager


def test_trainer_trains_with_parallel_producers():
    torch.manual_seed(0)
    rs = _rs()
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=32, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8, num_workers=2)
    assert gym._shared_params is not None                # trainer enabled sharing
    history = trainer.fit(total_steps=6, progress=False)
    assert len(history) == 6                             # 2 producer procs fed training
