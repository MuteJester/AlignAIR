import pytest
import torch
import torch.nn.functional as F

from alignair.aa.guard.losses import (energy_margin_loss, grad_reverse, guard_loss, supcon_loss)

pytestmark = pytest.mark.component


def test_grl_reverses_representation_grad_not_classifier_param_grad():
    # Safeguard 3: GRL reverses the gradient entering the shared ENCODER REP; the source-classifier params
    # get NORMAL gradients (they learn source prediction). GRL is identity in forward, so with the SAME
    # classifier + rep the loss value is unchanged -> classifier param grad identical, rep grad reversed.
    torch.manual_seed(0)
    rep = torch.randn(4, 8, requires_grad=True)
    sh = torch.nn.Linear(8, 3)
    tgt = torch.tensor([0, 1, 2, 0])
    loss0 = F.cross_entropy(sh(rep), tgt)
    g_rep0, g_w0 = torch.autograd.grad(loss0, [rep, sh.weight], retain_graph=True)
    loss1 = F.cross_entropy(sh(grad_reverse(rep, 1.0)), tgt)
    g_rep1, g_w1 = torch.autograd.grad(loss1, [rep, sh.weight])
    assert torch.allclose(g_rep1, -g_rep0, atol=1e-6)    # representation grad REVERSED
    assert torch.allclose(g_w1, g_w0, atol=1e-6)         # classifier param grad NORMAL (not reversed)


def test_energy_margin_penalizes_in_domain_above_m_in():
    energy = torch.tensor([8.0, 1.0])            # first is in-domain but energy > m_in(5) -> penalized
    is_out = torch.tensor([0, 0])
    mask = torch.tensor([1, 1])
    assert energy_margin_loss(energy, is_out, mask, m_in=5.0, m_out=10.0).item() > 0
    ok = energy_margin_loss(torch.tensor([1.0, 2.0]), is_out, mask, m_in=5.0)
    assert ok.item() == 0.0                      # both in-domain and below m_in -> no penalty


def test_energy_margin_penalizes_outlier_below_m_out():
    loss = energy_margin_loss(torch.tensor([3.0]), torch.tensor([1]), torch.tensor([1]), m_out=10.0)
    assert loss.item() > 0                        # outlier with energy 3 < m_out(10) -> penalized


def test_objective_masks_isolate_terms():
    # Safeguard 2: an all-zero mask contributes exactly zero (no invented labels), and only masked-in samples
    # drive their term.
    B = 4
    outputs = {"domain_type": torch.randn(B, 8, requires_grad=True),
               "completeness": torch.randn(B, 5),
               "applicability": torch.randn(B, 1),
               "source_logits": torch.randn(B, 3),
               "energy": torch.randn(B),
               "rep": torch.randn(B, 16)}
    targets = {"domain_type": torch.zeros(B), "completeness": torch.zeros(B),
               "applicability": torch.zeros(B), "source": torch.zeros(B),
               "energy_is_outlier": torch.zeros(B), "contrast": torch.zeros(B)}
    weights = {"complete": 1, "app": 1, "oe": 1, "contrast": 1, "source": 1}
    zero = torch.zeros(B)
    masks = {k: zero for k in ("type", "complete", "app", "energy", "contrast", "source")}
    total, parts = guard_loss(outputs, targets, weights, masks)
    assert float(total.detach()) == 0.0 and all(float(v.detach()) == 0.0 for v in parts.values())
    # turn on only the type objective -> only that term is non-trivial
    masks2 = dict(masks, type=torch.ones(B))
    _, parts2 = guard_loss(outputs, targets, weights, masks2)
    assert float(parts2["type"].detach()) > 0.0 and float(parts2["source"].detach()) == 0.0


def test_confounding_zero_source_mask_removes_source_term():
    # Safeguard 4: a stratum with confounded locus/source has source mask 0 -> source term contributes nothing
    B = 3
    outputs = {"domain_type": torch.randn(B, 8), "completeness": torch.randn(B, 5),
               "applicability": torch.randn(B, 1), "source_logits": torch.randn(B, 3),
               "energy": torch.randn(B), "rep": torch.randn(B, 16)}
    targets = {"domain_type": torch.zeros(B), "completeness": torch.zeros(B),
               "applicability": torch.zeros(B), "source": torch.zeros(B),
               "energy_is_outlier": torch.zeros(B), "contrast": torch.zeros(B)}
    masks = {k: torch.zeros(B) for k in ("type", "complete", "app", "energy", "contrast")}
    masks["source"] = torch.zeros(B)             # confounded -> masked out
    _, parts = guard_loss(outputs, targets, {}, masks)
    assert float(parts["source"]) == 0.0
