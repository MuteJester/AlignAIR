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


from alignair.nn.pointer_aligner import BandedPointerAligner, NEG


def _toy(B=2, S=8, Lg=20, d=16):
    torch.manual_seed(3)
    seg = torch.randn(B, S, d)
    germ = torch.randn(B, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool)
    gm = torch.ones(B, Lg, dtype=torch.bool)
    return seg, sm, germ, gm


def test_forward_shapes_and_masking():
    al = BandedPointerAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    gm[:, 15:] = False                                  # invalid germline tail
    sl, el = al(seg, sm, germ, gm)
    assert sl.shape == (2, 20) and el.shape == (2, 20)
    assert (sl[:, 15:] <= NEG + 1).all() and (el[:, 15:] <= NEG + 1).all()


def test_forward_localizes_planted_diagonal():
    # plant a high-cosine diagonal: seg rows == germ rows at offset 4. Tie the two
    # projections so identical reps project identically (what training converges to) ->
    # cosine peaks exactly on the true diagonal; tests the extraction+argmax wiring.
    al = BandedPointerAligner(d_model=16)
    al.germ_proj.load_state_dict(al.seg_proj.state_dict())
    B, S, Lg, d, off = 1, 6, 20, 16, 4
    g = torch.randn(B, Lg, d)
    seg = g[:, off:off + S, :].clone()                  # seg aligns to germ[off:off+S]
    sm = torch.ones(B, S, dtype=torch.bool)
    gm = torch.ones(B, Lg, dtype=torch.bool)
    sl, el = al(seg, sm, g, gm)
    assert sl.argmax(-1).item() == off
    assert el.argmax(-1).item() == off + S - 1


def test_alignment_score_shape_and_finite():
    al = BandedPointerAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    sc = al.alignment_score(seg, sm, germ, gm)
    assert sc.shape == (2,) and torch.isfinite(sc).all()


def test_novel_allele_floor_keeps_base_match_alive():
    # match_floor>0 means even with zeroed projections the base-match channel scores
    al = BandedPointerAligner(d_model=16, match_floor=1.0)
    with torch.no_grad():
        al.seg_proj.weight.zero_(); al.seg_proj.bias.zero_()
        al.germ_proj.weight.zero_(); al.germ_proj.bias.zero_()
    B, S, Lg = 1, 5, 12
    seg = torch.zeros(B, S, 16); germ = torch.zeros(B, Lg, 16)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    seg_tok = torch.tensor([[1, 2, 3, 4, 1]])
    germ_tok = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])
    sl, el = al(seg, sm, germ, gm, seg_tok=seg_tok, germ_tok=germ_tok)
    assert sl.argmax(-1).item() == 0                    # base match localizes start at 0
