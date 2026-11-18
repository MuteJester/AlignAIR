"""Trainer robustness — preflight rejection, grad-norm/clip, and (slow) validation/best/resume."""
import glob

import pytest
import torch

from alignair.core import AlignAIR, prepare_model
from alignair.train.config import TrainingConfig
from alignair.train.guards import TrainingConfigError
from alignair.train.records.streams import CurriculumStreamPolicy
from alignair.train.trainer import _train

pytestmark = pytest.mark.integration


def _bound(cfg, reference):
    """A chip model bound to ``reference`` (the chip reads its bank from it)."""
    model = AlignAIR(cfg)
    model.bind_reference(reference)
    return model


def _config(**fields):
    return TrainingConfig(dataconfigs=("HUMAN_IGH_OGRDB",), batch_size=4, lr=1e-3, **fields)


def test_a_bad_training_request_is_refused_when_its_config_is_built():
    """A bad request aborts up front (TrainingConfigError), before any reference or model is built."""
    with pytest.raises(TrainingConfigError, match="steps"):
        _config(steps=0)


@pytest.mark.slow
def test_resume_below_checkpoint_step_is_rejected(tmp_path):
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=576)
    ref, cfg, policy = setup.reference, setup.config, CurriculumStreamPolicy()
    torch.manual_seed(0)
    out = str(tmp_path / "m.alignair")
    _train(setup, _bound(cfg, ref), [gd.HUMAN_IGH_OGRDB], _config(steps=4), policy, save_path=out, save_every=100,
           log_every=100)
    with pytest.raises(ValueError, match="not beyond the checkpoint"):   # steps <= checkpoint step
        _train(setup, _bound(cfg, ref), [gd.HUMAN_IGH_OGRDB], _config(steps=2, resume_path=out), policy,
               save_path=out, log_every=100)


@pytest.fixture
def without_file_digest(monkeypatch):
    """Python 3.10, which the package supports, has no hashlib.file_digest (3.11+)."""
    import hashlib
    monkeypatch.delattr(hashlib, "file_digest", raising=False)


@pytest.mark.slow
def test_train_saves_best_and_resumes(tmp_path, without_file_digest):
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=576)
    ref, cfg, policy = setup.reference, setup.config, CurriculumStreamPolicy()
    torch.manual_seed(0)
    model = _bound(cfg, ref)
    out = str(tmp_path / "m.alignair")
    _train(setup, model, [gd.HUMAN_IGH_OGRDB], _config(steps=6, val_every=3, grad_clip=1.0), policy, save_path=out,
           save_every=3, val_batches=1, log_every=100)
    # best-checkpoint selection wrote <stem>.best.alignair
    assert glob.glob(str(tmp_path / "m.best.alignair"))
    # resume continues from the saved step (exercises RNG restore + state load)
    model2 = _bound(cfg, ref)
    _train(setup, model2, [gd.HUMAN_IGH_OGRDB], _config(steps=9, resume_path=out), policy, save_path=out,
           save_every=3, log_every=100)
    assert glob.glob(str(tmp_path / "m.step9.alignair"))
