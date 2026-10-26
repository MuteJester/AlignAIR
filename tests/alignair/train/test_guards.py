"""Training guards — preflight config validation and non-finite-loss abort."""
from types import SimpleNamespace

import pytest

from alignair.train.guards import (NonFiniteLossError, TrainingConfigError,
                                    check_finite_loss, validate_training_request)

pytestmark = pytest.mark.component


def _ref(nv=10, nj=5):
    return SimpleNamespace(gene=lambda g: {"V": [0] * nv, "J": [0] * nj}[g.upper()])


def _ok(**over):
    kw = dict(steps=100, batch_size=32, lr=3e-4, max_seq_length=576, reference=_ref())
    kw.update(over)
    return kw


def test_valid_request_passes():
    validate_training_request(**_ok())


@pytest.mark.parametrize("bad", [
    dict(steps=0), dict(steps=-1), dict(batch_size=0), dict(lr=0), dict(lr=-1e-3),
    dict(max_seq_length=0), dict(progresses=()), dict(progresses=(1.5,)), dict(heavy_shm=2.0),
    dict(short_boost=0), dict(grad_clip=0),
])
def test_invalid_request_raises(bad):
    with pytest.raises(TrainingConfigError):
        validate_training_request(**_ok(**bad))


def test_empty_reference_raises():
    with pytest.raises(TrainingConfigError, match="empty V"):
        validate_training_request(**_ok(reference=_ref(nv=0)))


def test_finite_loss_passes():
    check_finite_loss(10, 1.234, {"v_allele": 0.5})     # no raise


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_loss_aborts(bad):
    with pytest.raises(NonFiniteLossError, match="non-finite loss"):
        check_finite_loss(42, bad, {"v_allele": float("nan")})


def test_grad_clip_must_be_finite():
    with pytest.raises(TrainingConfigError, match="grad_clip"):
        validate_training_request(**_ok(grad_clip=float("inf")))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None],
                         ids=["nan-loss", "positive-infinite-loss", "negative-infinite-loss", "non-finite-gradient"])
def test_train_step_rejects_non_finite_loss_or_gradient_before_optimizer(monkeypatch, bad):
    """Invalid loss or gradient must abort the real training step before weights change."""
    import torch

    from alignair.core import losses
    from alignair.train.trainer import train_step

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.p

        def clamp_params(self):
            pass

    m = _M()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    # Inject numerical edge cases at the objective boundary, then exercise the
    # actual training step rather than duplicating its safety-check logic.
    monkeypatch.setattr(losses, "hierarchical_loss",
                        lambda out, *_: (torch.sqrt(out.abs()).sum() if bad is None
                                          else out.sum() + bad, {}))
    calls = []
    original_step = opt.step
    def optimizer_step(*args, **kwargs):
        calls.append(True)
        return original_step(*args, **kwargs)
    monkeypatch.setattr(opt, "step", optimizer_step)
    with pytest.raises(NonFiniteLossError, match="non-finite (loss|gradient)") as caught:
        train_step(m, {}, {}, object(), {}, opt, step=17)
    assert caught.value.code == "ALIGNAIR_TRAINING_NONFINITE"
    assert caught.value.context["stage"] == ("training_gradient" if bad is None else "training_loss")
    assert caught.value.context["step"] == 17
    assert calls == []
    torch.testing.assert_close(m.p.detach(), torch.zeros(1))
