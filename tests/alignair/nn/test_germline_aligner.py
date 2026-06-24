import torch
from alignair.nn.germline_aligner import GermlineAligner, decode_germline_coords


def test_aligner_output_shapes_and_germline_masking():
    d, B, Ls, Lg = 16, 2, 6, 10
    aligner = GermlineAligner(d_model=d)
    seg = torch.randn(B, Ls, d)
    seg_mask = torch.ones(B, Ls, dtype=torch.bool)
    germ = torch.randn(B, Lg, d)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    germ_mask[:, 7:] = False  # last 3 germline positions are padding
    sl, el = aligner(seg, seg_mask, germ, germ_mask)
    assert sl.shape == (B, Lg) and el.shape == (B, Lg)
    # masked germline positions never win the argmax
    gs, ge = decode_germline_coords(sl, el)
    assert (gs < 7).all() and (ge <= 7).all()


def test_decode_germline_coords_argmax():
    sl = torch.full((1, 8), -10.0)
    sl[0, 2] = 10.0
    el = torch.full((1, 8), -10.0)
    el[0, 6] = 10.0
    gs, ge = decode_germline_coords(sl, el)
    assert gs.tolist() == [2] and ge.tolist() == [7]  # end-exclusive = argmax+1


def test_soft_argmax_decode_centers_between_two_peaks():
    # two equal peaks at columns 4 and 6 -> soft-argmax ~5; argmax picks 4
    logits = torch.full((1, 11), -1e4)
    logits[0, 4] = 2.0
    logits[0, 6] = 2.0
    gs_hard, _ = decode_germline_coords(logits, logits, soft=False)
    gs_soft, _ = decode_germline_coords(logits, logits, soft=True)
    assert gs_hard.item() == 4
    assert gs_soft.item() == 5


def test_soft_decode_ignores_neg_masked_columns():
    logits = torch.full((1, 8), -1e4)
    logits[0, 2] = 5.0                      # only valid peak
    gs, ge = decode_germline_coords(logits, logits, soft=True)
    assert gs.item() == 2 and ge.item() == 3
