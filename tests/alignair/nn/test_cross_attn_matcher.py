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
