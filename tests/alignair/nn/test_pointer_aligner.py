import torch
from alignair.nn.aligner.diagonal_ops import weighted_leading_diag, weighted_reverse_diag


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


from alignair.nn.aligner.pointer import BandedPointerAligner, NEG


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


def test_banded_clean_diagonal_localizes_within_band():
    # A band with gamma=0 (init) tolerates offset shifts within +-G, so the clean-diagonal
    # start/end localize WITHIN the band of the true coords (gamma LEARNS to sharpen this in
    # training; at init the band is intentionally permissive). Tie projections so cosine peaks.
    from alignair.nn.aligner.pointer import BandedPointerAligner
    G = 3
    al = BandedPointerAligner(d_model=16, band_half_width=G)
    al.germ_proj.load_state_dict(al.seg_proj.state_dict())
    B, S, Lg, d, off = 1, 6, 20, 16, 8
    g = torch.randn(B, Lg, d)
    seg = g[:, off:off + S, :].clone()
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    sl, el = al(seg, sm, g, gm)
    assert abs(sl.argmax(-1).item() - off) <= G
    assert abs(el.argmax(-1).item() - (off + S - 1)) <= G


def test_banded_with_sharp_gamma_recovers_exact_localization():
    # When gamma strongly favors Delta=0, the band collapses to the single diagonal and
    # localizes EXACTLY -- confirming the band reduces to precise coords once trained.
    from alignair.nn.aligner.pointer import BandedPointerAligner
    G = 3
    al = BandedPointerAligner(d_model=16, band_half_width=G)
    al.germ_proj.load_state_dict(al.seg_proj.state_dict())
    with torch.no_grad():
        al.band_gamma.fill_(-50.0); al.band_gamma[G] = 0.0   # only Delta=0 survives
    B, S, Lg, d, off = 1, 6, 20, 16, 8
    g = torch.randn(B, Lg, d)
    seg = g[:, off:off + S, :].clone()
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    sl, el = al(seg, sm, g, gm)
    assert sl.argmax(-1).item() == off
    assert el.argmax(-1).item() == off + S - 1


def test_banded_start_end_reduces_to_single_diag_at_G0():
    from alignair.nn.aligner.diagonal_ops import banded_start_end
    torch.manual_seed(5)
    M = torch.randn(2, 5, 9); w = torch.rand(2, 5, 1)
    gamma = torch.zeros(1)
    s, e = banded_start_end(M, w, gamma, 0)
    assert torch.allclose(s, weighted_leading_diag(M, w), atol=1e-5)
    assert torch.allclose(e, weighted_reverse_diag(M, w), atol=1e-5)


def test_banded_aligner_registers_band_gamma():
    al = BandedPointerAligner(d_model=16, band_half_width=3)
    params = dict(al.named_parameters())
    assert "band_gamma" in params and params["band_gamma"].numel() == 7


def test_soft_argmax_localizes_edge_diagonal_precisely():
    # The clean-read coord regression was soft-argmax bias on a BROAD posterior truncated by
    # the germline edge (true start near column 0). A sharp posterior (high temp init) makes
    # the soft-argmax expected-position land ON the edge peak, not pulled inward.
    from alignair.nn.aligner.pointer import BandedPointerAligner
    from alignair.nn.aligner.germline_aligner import decode_germline_coords
    al = BandedPointerAligner(d_model=16)
    al.germ_proj.load_state_dict(al.seg_proj.state_dict())   # cosine peaks on the true diagonal
    B, S, Lg, d, off = 1, 8, 24, 16, 0                       # true start at the EDGE (col 0)
    g = torch.randn(B, Lg, d)
    seg = g[:, off:off + S, :].clone()
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    sl, el = al(seg, sm, g, gm)
    gs, ge = decode_germline_coords(sl, el, soft=True)
    assert gs.item() <= 1, f"soft-argmax start biased inward off the edge: {gs.item()}"
    assert abs(ge.item() - (off + S)) <= 1                   # end (exclusive) ~ off+S
