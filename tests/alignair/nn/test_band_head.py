import torch
from alignair.nn.aligner.band_head import BandHead, band_offset_loss, peak_evidence


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


def test_peak_evidence_overlap_fraction_separates_spurious():
    # overlap FRACTION: a real full-fit alignment ~1.0; a spurious peak near the germline END
    # covers only a few positions -> low fraction. This is what lets signal-absent reads fail open.
    B, S, Lg = 1, 20, 60
    seg_tok = torch.randint(1, 5, (B, S))
    germ_tok = torch.randint(1, 5, (B, Lg))
    germ_tok[0, 5:5 + S] = seg_tok[0]                     # match window at offset 5 (fits fully)
    sm = torch.ones(B, S, dtype=torch.bool)
    real = torch.full((B, Lg), -1e4); real[0, 5] = 10.0  # logits point at the REAL offset (fits)
    spurious = torch.full((B, Lg), -1e4); spurious[0, Lg - 3] = 10.0   # only 3 positions overlap
    ev_real = peak_evidence(real, seg_tok, germ_tok, sm)
    ev_spur = peak_evidence(spurious, seg_tok, germ_tok, sm)
    assert ev_real.item() == 1.0                          # all 20 read positions land on germline
    assert ev_spur.item() <= 3.0 / S + 1e-6               # only ~3 positions overlap -> tiny fraction
