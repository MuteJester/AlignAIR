import torch
from alignair.nn.aligner.banded_dp import band_mask_scores
from alignair.nn.aligner.soft_dp import NEG


def test_band_mask_keeps_band_drops_rest():
    B, S, Lg = 1, 4, 20
    M = torch.zeros(B, S, Lg)
    center = torch.tensor([5])
    out = band_mask_scores(M, center, w=2)
    # row i=0: keep |j-5|<=2 -> cols 3..7 ; row i=1: center 6 -> cols 4..8
    assert (out[0, 0, 3:8] == 0.0).all()
    assert out[0, 0, 2] <= NEG + 1 and out[0, 0, 8] <= NEG + 1
    assert (out[0, 1, 4:9] == 0.0).all()


def test_full_width_band_is_noop():
    B, S, Lg = 2, 5, 12
    M = torch.randn(B, S, Lg)
    center = torch.zeros(B, dtype=torch.long)
    out = band_mask_scores(M, center, w=Lg)        # band covers everything
    assert torch.equal(out, M)


from alignair.nn.aligner.banded_dp import SeedExtendAligner
from alignair.nn.aligner.soft_dp import SoftDPAligner


def _toy(B=2, S=10, Lg=24, d=16):
    torch.manual_seed(0)
    return (torch.randn(B, S, d), torch.ones(B, S, dtype=torch.bool),
            torch.randn(B, Lg, d), torch.ones(B, Lg, dtype=torch.bool))


def _copy_params(al, sd):
    al.seg_proj.load_state_dict(sd.seg_proj.state_dict())
    al.germ_proj.load_state_dict(sd.germ_proj.state_dict())
    with torch.no_grad():
        for p in ("log_scale", "_gap_open", "_gap_extend", "_del_gap", "_match_weight"):
            getattr(al, p).copy_(getattr(sd, p))


def test_forward_shapes():
    al = SeedExtendAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    sl, el = al(seg, sm, germ, gm, center, w=8)
    assert sl.shape == (2, 24) and el.shape == (2, 24)


def test_full_band_matches_softdp_within_tol():
    sd = SoftDPAligner(d_model=16)
    al = SeedExtendAligner(d_model=16)
    _copy_params(al, sd)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    sl_a, el_a = al(seg, sm, germ, gm, center, w=germ.shape[1])    # full band, tokens=None -> pure cosine
    sl_s, el_s = sd(seg, sm, germ, gm)
    assert torch.allclose(el_a, el_s, atol=1e-3, rtol=1e-3)
    assert torch.allclose(sl_a, sl_s, atol=1e-3, rtol=1e-3)


def test_base_match_changes_output():
    al = SeedExtendAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    st = torch.randint(1, 5, (2, 10)); gt = torch.randint(1, 5, (2, 24))
    _, el_plain = al(seg, sm, germ, gm, center, w=24)
    _, el_bm = al(seg, sm, germ, gm, center, w=24, seg_tok=st, germ_tok=gt)
    assert not torch.allclose(el_plain, el_bm)                     # base-match is a live input


def test_alignment_score_full_band_matches_softdp():
    sd = SoftDPAligner(d_model=16)
    al = SeedExtendAligner(d_model=16)
    _copy_params(al, sd)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    st = torch.randint(1, 5, (2, 10)); gt = torch.randint(1, 5, (2, 24))
    a = al.alignment_score(seg, sm, germ, gm, center, w=24, seg_tok=st, germ_tok=gt)
    s = sd.alignment_score(seg, sm, germ, gm, seg_tok=st, germ_tok=gt)
    assert a.shape == (2,) and torch.allclose(a, s, atol=1e-3, rtol=1e-3)
