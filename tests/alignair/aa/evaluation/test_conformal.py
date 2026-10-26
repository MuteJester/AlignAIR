import pytest
import numpy as np

from alignair.aa.conformal import Conformal, aps_scores, aps_set, conformal_quantile

pytestmark = pytest.mark.component


def _synthetic(n, C, seed):
    """Well-specified softmax outputs whose argmax is the true label with realistic noise."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, C))
    true = rng.integers(0, C, size=n)
    logits[np.arange(n), true] += 2.0                         # signal so the true class is usually high
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p, true


def test_aps_gives_marginal_coverage_near_target():
    pc, tc = _synthetic(4000, 20, 0)
    pt, tt = _synthetic(4000, 20, 1)
    cal = Conformal(alpha=0.1, seed=0).fit(pc, tc)
    covered = [tt[i] in cal.predict_set(pt[i]) for i in range(len(tt))]
    cov = np.mean(covered)
    assert 0.86 <= cov <= 0.94                                # ~90% target, finite-sample slack


def test_smaller_alpha_gives_larger_sets_and_higher_coverage():
    pc, tc = _synthetic(4000, 20, 2)
    pt, tt = _synthetic(4000, 20, 3)
    c90 = Conformal(alpha=0.1, seed=0).fit(pc, tc)
    c99 = Conformal(alpha=0.01, seed=0).fit(pc, tc)
    s90 = np.mean([len(c90.predict_set(pt[i])) for i in range(len(tt))])
    s99 = np.mean([len(c99.predict_set(pt[i])) for i in range(len(tt))])
    cov90 = np.mean([tt[i] in c90.predict_set(pt[i]) for i in range(len(tt))])
    cov99 = np.mean([tt[i] in c99.predict_set(pt[i]) for i in range(len(tt))])
    assert s99 > s90 and cov99 > cov90


def test_raps_option_maintains_coverage():
    # RAPS is an available option; its size-reduction is tuning-dependent, but coverage must hold.
    pc, tc = _synthetic(4000, 30, 4)
    pt, tt = _synthetic(4000, 30, 5)
    raps = Conformal(alpha=0.1, lam=0.1, k_reg=2, seed=0).fit(pc, tc)
    cov = np.mean([tt[i] in raps.predict_set(pt[i]) for i in range(len(tt))])
    assert 0.86 <= cov <= 0.96


def test_mondrian_uses_group_quantile_when_big_enough_else_pooled():
    pc, tc = _synthetic(2000, 10, 6)
    groups = np.array(["big"] * 1500 + ["tiny"] * 500)
    c = Conformal(alpha=0.1, min_group=1000, seed=0).fit(pc, tc, groups=groups)
    assert "big" in c.q_group and "tiny" not in c.q_group     # tiny -> pooled fallback
