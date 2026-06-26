import math
import torch
from alignair.nn.aligner.base_match import base_match_channel


def test_passthrough_without_tokens():
    M = torch.randn(2, 4, 6)
    out = base_match_channel(M, None, None, None, torch.tensor(1.0), 1.0)
    assert torch.equal(out, M)


def test_matches_softdp_scores_exactly():
    # identical to SoftDPAligner._scores base-match math (no reliability)
    torch.manual_seed(0)
    B, S, Lg = 2, 5, 7
    M = torch.randn(B, S, Lg)
    seg_tok = torch.randint(0, 5, (B, S))
    germ_tok = torch.randint(0, 5, (B, Lg))
    mw, floor = torch.tensor(1.0), 1.0
    out = base_match_channel(M, seg_tok, germ_tok, None, mw, floor)
    # reference: the exact lines from soft_dp_aligner._scores
    st, gt = seg_tok.unsqueeze(2), germ_tok.unsqueeze(1)
    real = (st >= 1) & (st <= 4) & (gt >= 1) & (gt <= 4)
    u = real.float() * (2.0 * (st == gt).float() - 1.0)
    lam = floor + torch.nn.functional.softplus(mw)
    ref = M + lam * u
    assert torch.allclose(out, ref, atol=1e-6)


def test_reliability_gates_term_toward_zero():
    # reliability 0 -> base-match contribution collapses (a*u + norm term ~ 0)
    B, S, Lg = 1, 4, 5
    M = torch.zeros(B, S, Lg)
    seg_tok = torch.tensor([[1, 2, 3, 4]])
    germ_tok = torch.tensor([[1, 2, 3, 4, 1]])
    rel0 = torch.zeros(B, S)
    out0 = base_match_channel(M, seg_tok, germ_tok, rel0, torch.tensor(1.0), 1.0)
    # with a=0: a*u = 0 and norm = log(1+3)-log(4) = 0 -> M unchanged
    assert torch.allclose(out0, M, atol=1e-6)


def test_base_match_matrix_signs():
    from alignair.nn.aligner.base_match import base_match_matrix
    seg = torch.tensor([[1, 2, 0]])        # A, C, pad
    germ = torch.tensor([[1, 1, 2]])       # A, A, C
    M = base_match_matrix(seg, germ)
    assert M.shape == (1, 3, 3)
    assert M[0, 0, 0] == 1.0 and M[0, 0, 1] == 1.0 and M[0, 0, 2] == -1.0
    assert M[0, 1, 0] == -1.0
    assert (M[0, 2] == 0.0).all()           # pad token (0) -> 0 everywhere
