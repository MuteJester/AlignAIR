import pytest
from alignair.aa.guard.model import GuardModel

pytestmark = pytest.mark.component


def test_guardmodel_forward_shapes_and_c1_needs_no_fusion():
    import torch
    m = GuardModel("C4", hmm_dim=6, width=64, depth=2, heads=4, n_sources=3)
    ids = torch.zeros(2, 10, dtype=torch.long); mask = torch.ones(2, 10, dtype=torch.bool)
    out = m(ids, mask, hmm_norm=torch.zeros(2, 6))
    assert out["applicability"].shape == (2, 1) and out["domain_type"].shape == (2, 8)
    assert out["source_logits"].shape == (2, 3) and out["energy"].shape == (2,)
    m1 = GuardModel("C1", hmm_dim=6, width=64, depth=2)
    o1 = m1(ids, mask)                                    # C1: no fusion required
    assert o1["source_logits"].shape == (2, 1)
