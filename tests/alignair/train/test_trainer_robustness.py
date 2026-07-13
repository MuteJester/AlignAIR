"""P0-10: trainer robustness — preflight rejection, grad-norm/clip, and (slow) validation/best/resume."""
import glob
from types import SimpleNamespace

import pytest
import torch

from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.core.losses import make_logvars
from alignair.train.guards import TrainingConfigError
from alignair.train.trainer import train


def _tiny_model():
    cfg = AlignAIRConfig(max_seq_length=576, has_d=True,
                         v_allele_count=4, d_allele_count=3, j_allele_count=2)
    return cfg, AlignAIR(cfg), make_logvars(cfg)


def test_train_rejects_bad_config_before_training():
    """A bad request aborts up front (TrainingConfigError) — no long run on invalid hyperparameters."""
    cfg, model, logvars = _tiny_model()
    ref = SimpleNamespace(gene=lambda g: [0] * 4)          # non-empty V/J
    with pytest.raises(TrainingConfigError, match="steps"):
        train(model, ref, [object()], cfg, logvars, steps=0)     # steps=0 -> invalid, raises immediately


def test_train_rejects_empty_reference():
    cfg, model, logvars = _tiny_model()
    ref = SimpleNamespace(gene=lambda g: [])               # empty V/J allele set
    with pytest.raises(TrainingConfigError, match="empty V"):
        train(model, ref, [object()], cfg, logvars, steps=10)


@pytest.mark.slow
def test_train_saves_best_and_resumes(tmp_path):
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cfg = AlignAIRConfig(max_seq_length=576, has_d=True,
                         v_allele_count=len(ref.gene("V")), j_allele_count=len(ref.gene("J")),
                         d_allele_count=len(ref.gene("D")))
    torch.manual_seed(0)
    model, logvars = AlignAIR(cfg), make_logvars(cfg)
    out = str(tmp_path / "m.alignair")
    train(model, ref, [gd.HUMAN_IGH_OGRDB], cfg, logvars, steps=6, batch_size=4, lr=1e-3,
          save_path=out, save_every=3, val_every=3, val_batches=1, grad_clip=1.0, log_every=100)
    # best-checkpoint selection wrote <stem>.best.alignair
    assert glob.glob(str(tmp_path / "m.best.alignair"))
    # resume continues from the saved step (exercises RNG restore + state load)
    model2, logvars2 = AlignAIR(cfg), make_logvars(cfg)
    train(model2, ref, [gd.HUMAN_IGH_OGRDB], cfg, logvars2, steps=9, batch_size=4, lr=1e-3,
          save_path=out, resume_path=out, save_every=3, log_every=100)
    assert glob.glob(str(tmp_path / "m.step9.alignair"))
