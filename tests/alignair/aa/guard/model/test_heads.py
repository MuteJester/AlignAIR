import pytest
import torch

from alignair.aa.guard.heads import GuardHeads

pytestmark = pytest.mark.component


def test_head_logit_shapes():
    h = GuardHeads(in_dim=32)
    o = h(torch.randn(4, 32))
    assert o["completeness"].shape == (4, 5)
    assert o["domain_type"].shape == (4, 8)
    assert o["applicability"].shape == (4, 1)


def test_energy_is_neg_logsumexp_and_decreases_as_known_type_grows():
    logits = torch.zeros(1, 8)
    e0 = GuardHeads.energy(logits)
    assert torch.allclose(e0, -torch.logsumexp(logits, dim=-1))
    grown = logits.clone(); grown[0, 3] = 5.0          # a recognized-type logit grows
    e1 = GuardHeads.energy(grown)
    assert bool((e1 < e0).all())                       # energy decreases
