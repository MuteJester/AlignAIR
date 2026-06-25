import torch
from alignair.nn.band_head import base_match_matrix, BandHead, band_offset_loss


def test_base_match_matrix_signs():
    seg = torch.tensor([[1, 2, 0]])        # A, C, pad
    germ = torch.tensor([[1, 1, 2]])       # A, A, C
    M = base_match_matrix(seg, germ)
    assert M.shape == (1, 3, 3)
    assert M[0, 0, 0] == 1.0                # A vs A match
    assert M[0, 0, 1] == 1.0                # A vs A match
    assert M[0, 0, 2] == -1.0               # A vs C mismatch
    assert M[0, 1, 0] == -1.0               # C vs A mismatch
    assert (M[0, 2] == 0.0).all()           # pad token (0) -> 0 everywhere


def test_bandhead_localizes_clean_offset():
    # base-match alone (zeroed cosine projections) must localize the true offset
    al = BandHead(d_model=16)
    with torch.no_grad():
        al.proj_s.weight.zero_(); al.proj_s.bias.zero_()
        al.proj_g.weight.zero_(); al.proj_g.bias.zero_()
    B, S, Lg, off = 1, 8, 30, 7
    seg_tok = torch.randint(1, 5, (B, S))
    germ_tok = torch.randint(1, 5, (B, Lg))
    germ_tok[0, off:off + S] = seg_tok[0]                 # plant exact match at offset 7
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    seg = torch.zeros(B, S, 16); germ = torch.zeros(B, Lg, 16)
    logits = al(seg, sm, germ, gm, seg_tok, germ_tok)
    assert logits.shape == (B, Lg)
    assert logits.argmax(-1).item() == off


def test_bandhead_masks_invalid_columns():
    al = BandHead(d_model=16)
    B, S, Lg = 2, 6, 20
    seg = torch.randn(B, S, 16); germ = torch.randn(B, Lg, 16)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    gm[:, 15:] = False
    st = torch.randint(1, 5, (B, S)); gt = torch.randint(1, 5, (B, Lg))
    logits = al(seg, sm, germ, gm, st, gt)
    assert (logits[:, 15:] <= -1e3).all()


def test_band_offset_loss_minimized_at_truth():
    B, Lg = 4, 30
    true = torch.tensor([5, 10, 3, 12])
    pos = torch.arange(Lg).float()
    good = -(pos[None] - true[:, None].float()) ** 2          # peaked on truth
    bad = -(pos[None] - (true[:, None].float() + 8)) ** 2
    assert band_offset_loss(good, true) < band_offset_loss(bad, true)
