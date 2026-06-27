import torch
from alignair.nn.heads.cross_attn_matcher import CrossAttnMatcher


def _toy(B=2, C=3, S=5, Lg=7, d=16):
    torch.manual_seed(0)
    seg = torch.randn(B, S, d)
    cand = torch.randn(B, C, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool)
    cm = torch.ones(B, C, Lg, dtype=torch.bool)
    return seg, sm, cand, cm


def test_forward_shapes():
    m = CrossAttnMatcher(d_model=16, nhead=4)
    seg, sm, cand, cm = _toy()
    match, gs, ge = m(seg, sm, cand, cm)
    assert match.shape == (2, 3)
    assert gs.shape == (2, 3, 7) and ge.shape == (2, 3, 7)


def test_matching_candidate_scores_highest():
    # candidate 1's germline tokens ARE the segment tokens (a perfect match); the matcher
    # should score candidate 1 above the two random candidates. (Untrained projections, so we
    # assert it on the raw alignment signal by initialising q/k/v near-identity.)
    torch.manual_seed(0)
    m = CrossAttnMatcher(d_model=16, nhead=4)
    with torch.no_grad():                                  # near-identity q/k so the match signal shows
        for lin in (m.q, m.k):
            lin.weight.copy_(torch.eye(16)); lin.bias.zero_()
    B, C, S, Lg, d = 1, 3, 6, 6, 16
    seg = torch.randn(B, S, d)
    cand = torch.randn(B, C, Lg, d)
    cand[0, 1] = seg[0]                                    # candidate 1 == the segment
    sm = torch.ones(B, S, dtype=torch.bool)
    cm = torch.ones(B, C, Lg, dtype=torch.bool)
    match, _, _ = m(seg, sm, cand, cm)
    assert match.argmax(dim=-1).item() == 1


def test_germline_pointers_track_a_shifted_match():
    # segment of length 4 equals germline positions 2..6 of candidate 0 (a 5' germline trim of 2).
    # The first seg token should point near germline pos 2 (start); the last near pos 5 (end).
    torch.manual_seed(0)
    m = CrossAttnMatcher(d_model=16, nhead=4)
    with torch.no_grad():
        for lin in (m.q, m.k):
            lin.weight.copy_(torch.eye(16)); lin.bias.zero_()
    B, C, S, Lg, d = 1, 1, 4, 8, 16
    germ = torch.randn(B, C, Lg, d)
    seg = germ[0, 0, 2:6].clone().unsqueeze(0)             # seg == germline[2:6]
    sm = torch.ones(B, S, dtype=torch.bool)
    cm = torch.ones(B, C, Lg, dtype=torch.bool)
    _, gs, ge = m(seg, sm, germ, cm)
    assert gs[0, 0].argmax().item() == 2                  # start pointer -> germline pos 2
    assert ge[0, 0].argmax().item() == 5                  # end pointer -> germline pos 5


def test_pointer_logits_respect_germline_mask():
    m = CrossAttnMatcher(d_model=16, nhead=4)
    seg, sm, cand, cm = _toy()
    cm[:, :, -2:] = False                                  # mask last 2 germline positions
    _, gs, ge = m(seg, sm, cand, cm)
    assert (gs[..., -2:] <= -1e8).all()                   # masked germline positions -> -inf
    assert (ge[..., -2:] <= -1e8).all()
