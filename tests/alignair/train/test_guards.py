"""P0-10: training guards — preflight config validation and non-finite-loss abort."""
from types import SimpleNamespace

import pytest

from alignair.train.guards import (NonFiniteLossError, TrainingConfigError,
                                    check_finite_loss, validate_training_request)


def _ref(nv=10, nj=5):
    return SimpleNamespace(gene=lambda g: SimpleNamespace(__len__=lambda: 0) if False
                           else {"V": [0] * nv, "J": [0] * nj}[g.upper()])


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
