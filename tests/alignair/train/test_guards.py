"""Training guards — preflight config validation and non-finite-loss abort."""
from types import SimpleNamespace

import pytest

from alignair.errors import ConfigurationError
from alignair.train.config import TrainingConfig
from alignair.train.guards import (NonFiniteTrainingError, TrainingConfigError, check_finite_loss,
                                   validate_training_reference, validate_training_request)
from alignair.train.records.streams import CurriculumStreamPolicy

pytestmark = pytest.mark.component


def _ref(nv=10, nj=5):
    return SimpleNamespace(gene=lambda g: {"V": [0] * nv, "J": [0] * nj}[g.upper()])


def _ok(**over):
    kw = dict(steps=100, batch_size=32, lr=3e-4)
    kw.update(over)
    return kw


def test_valid_request_passes():
    validate_training_request(**_ok())
    validate_training_reference(reference=_ref(), max_seq_length=576)


@pytest.mark.parametrize("bad", [
    dict(steps=0), dict(steps=-1), dict(steps=True), dict(batch_size=0), dict(lr=0), dict(lr=-1e-3),
    dict(lr=float("nan")), dict(grad_clip=0),
])
def test_invalid_request_raises(bad):
    with pytest.raises(TrainingConfigError):
        validate_training_request(**_ok(**bad))


@pytest.mark.parametrize("bad", [dict(steps=0), dict(batch_size=0), dict(lr=0), dict(grad_clip=0)])
def test_a_training_config_refuses_a_bad_request_when_it_is_built(bad):
    with pytest.raises(TrainingConfigError):
        TrainingConfig(dataconfigs=("HUMAN_IGH_OGRDB",), **bad)


@pytest.mark.parametrize("bad", [dict(progresses=()), dict(progresses=(1.5,)), dict(heavy_shm=2.0),
                                 dict(short_boost=0)])
def test_the_record_policy_refuses_its_bad_values(bad):
    with pytest.raises(ConfigurationError):
        CurriculumStreamPolicy(**bad)


def test_a_bad_window_raises():
    with pytest.raises(TrainingConfigError, match="max_seq_length"):
        validate_training_reference(reference=_ref(), max_seq_length=0)


def test_empty_reference_raises():
    with pytest.raises(TrainingConfigError, match="empty V"):
        validate_training_reference(reference=_ref(nv=0), max_seq_length=576)


def test_finite_loss_passes():
    check_finite_loss(10, 1.234, {"v_allele": 0.5})     # no raise


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_loss_aborts(bad):
    with pytest.raises(NonFiniteTrainingError, match="non-finite loss"):
        check_finite_loss(42, bad, {"v_allele": float("nan")})


def test_grad_clip_must_be_finite():
    with pytest.raises(TrainingConfigError, match="grad_clip"):
        validate_training_request(**_ok(grad_clip=float("inf")))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None],
                         ids=["nan-loss", "positive-infinite-loss", "negative-infinite-loss", "non-finite-gradient"])
def test_a_training_step_rejects_non_finite_loss_or_gradient_before_optimizer(monkeypatch, bad):
    """Invalid loss or gradient must abort the real training step before weights change."""
    import torch

    from alignair.train.engine import TrainingEngine

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.p

    m = _M()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    # Inject numerical edge cases at the objective boundary, then exercise the
    # actual training step rather than duplicating its safety-check logic.
    tasks = SimpleNamespace(loss=lambda out, *_: (
        torch.sqrt(out.abs()).sum() if bad is None else out.sum() + bad, {}))
    calls = []
    original_step = opt.step
    def optimizer_step(*args, **kwargs):
        calls.append(True)
        return original_step(*args, **kwargs)
    monkeypatch.setattr(opt, "step", optimizer_step)
    with pytest.raises(NonFiniteTrainingError, match="non-finite (loss|gradient)") as caught:
        TrainingEngine(m, opt, tasks=tasks).step(({}, {}), step=17)
    assert caught.value.code == "ALIGNAIR_TRAINING_NUMERICS_INVALID"
    assert caught.value.context["stage"] == ("training_gradient" if bad is None else "training_loss")
    assert caught.value.context["step"] == 17
    assert calls == []
    torch.testing.assert_close(m.p.detach(), torch.zeros(1))
