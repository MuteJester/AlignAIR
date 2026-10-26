"""Positive-only scorer candidates: fit on positives only, score in-manifold > out, pickle-free round-trip."""
from __future__ import annotations

import numpy as np
import pytest

from alignair.aa.product.applicability.scorers import (
    ShrinkageMahalanobisScorer, KNNScorer, CANDIDATES)

pytestmark = pytest.mark.component


def _data():
    rng = np.random.RandomState(0)
    # in-manifold cluster (fit is positive-only), and a distant out-of-manifold cluster
    d = 40
    fit = rng.normal(0, 1, size=(200, d))
    in_test = rng.normal(0, 1, size=(50, d))
    ood = rng.normal(6, 1, size=(50, d))
    return fit, in_test, ood


@pytest.mark.parametrize("Cls", [ShrinkageMahalanobisScorer, KNNScorer])
def test_in_manifold_scores_higher_than_ood(Cls):
    fit, in_test, ood = _data()
    s = Cls().fit(fit)                                   # positive-only fit
    assert s.score(in_test).mean() > s.score(ood).mean()


@pytest.mark.parametrize("name,Cls", list(CANDIDATES.items()))
def test_serialization_is_pickle_free_and_exact(name, Cls):
    fit, in_test, _ = _data()
    s = Cls().fit(fit)
    blob = s.to_bytes()
    assert b"__reduce__" not in blob and b"cpickle" not in blob and b"numpy.core.multiarray" not in blob
    s2 = Cls.from_bytes(blob)
    assert np.allclose(s.score(in_test), s2.score(in_test))


def test_mahalanobis_handles_dim_exceeding_fit_count():
    # d > n: sample covariance is singular; Ledoit-Wolf shrinkage must keep it invertible
    rng = np.random.RandomState(1)
    fit = rng.normal(0, 1, size=(30, 120))
    s = ShrinkageMahalanobisScorer().fit(fit)
    assert np.isfinite(s.score(fit)).all()
