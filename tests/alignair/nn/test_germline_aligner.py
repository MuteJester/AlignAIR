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
