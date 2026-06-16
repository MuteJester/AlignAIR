import torch
from alignair.losses.dnalignair_loss import DNAlignAIRLoss


def _outputs(B, L, nV, nJ, nD):
    return {
        "orientation_logits": torch.randn(B, 4, requires_grad=True),
        "region_logits": torch.randn(B, L, 8, requires_grad=True),
        "state_logits": torch.randn(B, L, 4, requires_grad=True),
        "noise_count": torch.rand(B, 1, requires_grad=True),
        "mutation_rate": torch.rand(B, 1, requires_grad=True),
        "indel_count": torch.rand(B, 1, requires_grad=True),
        "productive": torch.rand(B, 1, requires_grad=True),
        "match": {"V": torch.randn(B, nV, requires_grad=True),
                  "J": torch.randn(B, nJ, requires_grad=True),
                  "D": torch.randn(B, nD, requires_grad=True)},
    }


def _batch(B, L, nV, nJ, nD):
    region = torch.randint(0, 8, (B, L)); region[:, L // 2:] = -100
    state = torch.randint(0, 4, (B, L)); state[:, L // 2:] = -100
    vA = torch.zeros(B, nV); vA[:, 0] = 1.0
    jA = torch.zeros(B, nJ); jA[:, 0] = 1.0
    dA = torch.zeros(B, nD); dA[:, 0] = 1.0
    return {
        "orientation_id": torch.zeros(B, dtype=torch.long),
        "region_labels": region, "state_labels": state,
        "noise_count": torch.rand(B, 1), "mutation_rate": torch.rand(B, 1),
        "indel_count": torch.rand(B, 1), "productive": torch.ones(B, 1),
        "v_allele": vA, "j_allele": jA, "d_allele": dA,
    }


def test_loss_finite_backprops_and_has_components():
    B, L, nV, nJ, nD = 3, 12, 5, 3, 4
    loss_fn = DNAlignAIRLoss(has_d=True)
    out, batch = _outputs(B, L, nV, nJ, nD), _batch(B, L, nV, nJ, nD)
    total, comp = loss_fn(out, batch)
    assert torch.isfinite(total)
    total.backward()
    assert out["region_logits"].grad is not None
    for k in ("orientation", "region", "state", "v_match", "j_match", "d_match",
              "noise", "mutation", "indel", "productive"):
        assert k in comp


def test_loss_no_d_omits_d_match():
    B, L, nV, nJ = 2, 10, 4, 2
    loss_fn = DNAlignAIRLoss(has_d=False)
    out = _outputs(B, L, nV, nJ, 1)
    del out["match"]["D"]
    batch = _batch(B, L, nV, nJ, 1)
    total, comp = loss_fn(out, batch)
    assert "d_match" not in comp and torch.isfinite(total)
