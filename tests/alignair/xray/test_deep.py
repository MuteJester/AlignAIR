"""Tests for the deep xray tiers: geometry, dynamics/interference, deep_probe."""
import torch

from alignair.xray import deep_probe, dynamics, geometry


# ---------------- geometry ----------------
def test_effective_rank_and_cka():
    rank1 = torch.outer(torch.arange(1.0, 21), torch.arange(1.0, 6))     # [20,5] rank-1
    assert geometry.effective_rank(rank1) < 1.5                          # ~1 dimension used
    x = torch.randn(64, 16)
    assert geometry.effective_rank(x) > 8                                # full-ish rank
    assert abs(geometry.linear_cka(x, x) - 1.0) < 1e-4                   # self-similarity = 1


def test_feature_collinearity_high_for_duplicated():
    c = torch.randn(64, 1)
    dup = c.repeat(1, 5)                                                 # 5 identical columns
    indep = torch.randn(64, 5)
    assert geometry.feature_collinearity(dup) > geometry.feature_collinearity(indep) + 0.5


# ---------------- dynamics / interference ----------------
def test_gradient_conflict_detects_opposition():
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    la = (w * torch.tensor([1.0, 1.0])).sum()                           # grad [1,1]
    lb = (w * torch.tensor([-1.0, -1.0])).sum()                         # grad [-1,-1] -> opposed
    res = dynamics.gradient_conflict({"a": la, "b": lb}, [w])
    assert abs(res["cosine"]["a|b"] + 1.0) < 1e-5 and res["min_cosine"] < -0.99


def test_weight_velocity():
    m = torch.nn.Linear(4, 4)
    snap = dynamics.weight_snapshot(m)
    with torch.no_grad():
        m.weight += 0.1
    vel = dynamics.weight_velocity(m, snap)
    assert vel["weight"] > 0


# ---------------- deep_probe ----------------
def test_hessian_top_eigenvalue_quadratic():
    w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    a = torch.tensor([3.0, 1.0])

    def loss_closure():
        return 0.5 * (a * w * w).sum()                                  # Hessian = diag(3,1)

    assert abs(deep_probe.hessian_top_eigenvalue(loss_closure, [w], iters=40) - 3.0) < 0.1


def test_neural_collapse_separated_vs_overlap():
    torch.manual_seed(0)
    labels = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    sep = torch.cat([torch.randn(20, 4) * 0.01 + 5, torch.randn(20, 4) * 0.01 - 5])
    overlap = torch.cat([torch.randn(20, 4) * 2, torch.randn(20, 4) * 2])
    assert (deep_probe.neural_collapse(sep, labels)["nc1_within_between"]
            < deep_probe.neural_collapse(overlap, labels)["nc1_within_between"])


def test_weightwatcher_alpha_and_margin():
    m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 32))
    alphas = deep_probe.weightwatcher_alpha(m)
    assert len(alphas) >= 1 and all(a > 0 for a in alphas.values())
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
    mrg = deep_probe.classification_margin(logits, torch.tensor([0, 1]))
    assert mrg["mean"] > 0
