import torch
from alignair.nn.pointer_aligner import weighted_leading_diag, weighted_reverse_diag


def _ref_leading(M, w):
    B, S, Lg = M.shape
    out = torch.zeros(B, Lg)
    for b in range(B):
        denom = w[b, :, 0].sum().clamp(min=1e-6)
        for o in range(Lg):
            s = 0.0
            for i in range(S):
                if o + i < Lg:
                    s += w[b, i, 0] * M[b, i, o + i]
            out[b, o] = s / denom
    return out


def _ref_reverse(M, w):
    B, S, Lg = M.shape
    out = torch.zeros(B, Lg)
    for b in range(B):
        denom = w[b, :, 0].sum().clamp(min=1e-6)
        for o in range(Lg):
            s = 0.0
            for i in range(S):
                j = o - i
                if 0 <= j < Lg:
                    s += w[b, S - 1 - i, 0] * M[b, S - 1 - i, j]
            out[b, o] = s / denom
    return out


def test_leading_diag_matches_reference_nonuniform_w():
    torch.manual_seed(1)
    B, S, Lg = 2, 5, 7
    M = torch.randn(B, S, Lg)
    w = torch.rand(B, S, 1)                       # NON-UNIFORM (mandatory)
    assert torch.allclose(weighted_leading_diag(M, w), _ref_leading(M, w), atol=1e-5)


def test_reverse_diag_matches_reference_nonuniform_w():
    torch.manual_seed(2)
    B, S, Lg = 2, 5, 7
    M = torch.randn(B, S, Lg)
    w = torch.rand(B, S, 1)                       # NON-UNIFORM (catches the flip-w bug)
    assert torch.allclose(weighted_reverse_diag(M, w), _ref_reverse(M, w), atol=1e-5)


def test_diag_helpers_are_autograd_safe():
    M = torch.randn(1, 4, 6, requires_grad=True)
    w = torch.rand(1, 4, 1)
    (weighted_leading_diag(M, w).sum() + weighted_reverse_diag(M, w).sum()).backward()
    assert M.grad is not None and torch.isfinite(M.grad).all()
