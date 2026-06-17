import torch

from alignair.nn.soft_dp_aligner import soft_dp_end_logits, SoftDPAligner, NEG


def _gaps(go=-2.0, ge=-1.0, dg=-1.0):
    t = lambda v: torch.tensor(v)
    return t(go), t(ge), t(dg)


def test_exact_contiguous_match_end_localizes():
    # read of length S maps contiguously to germline columns [c0, c0+S)
    B, S, Lg, c0 = 1, 10, 30, 5
    M = torch.full((B, S, Lg), 0.0)
    for i in range(S):
        M[0, i, c0 + i] = 6.0                       # high score on the alignment diagonal
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    end = soft_dp_end_logits(M, seg_mask, germ_mask, *_gaps())
    assert end.argmax(-1).item() == c0 + S - 1     # last matched germline column = 14


def test_deletion_in_germline_still_localizes_end():
    # read 0-4 -> germ 5-9, germ 10-11 DELETED (skipped), read 5-9 -> germ 12-16
    B, S, Lg = 1, 10, 30
    M = torch.full((B, S, Lg), 0.0)
    mapping = [5, 6, 7, 8, 9, 12, 13, 14, 15, 16]   # 2-column germline deletion in the middle
    for i, j in enumerate(mapping):
        M[0, i, j] = 6.0
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    end = soft_dp_end_logits(M, seg_mask, germ_mask, *_gaps())
    assert end.argmax(-1).item() == 16             # spans the deletion (diagonal corr would fail)


def test_variable_seg_len_uses_correct_end_row():
    # sample 0 has 6 valid read positions, sample 1 has 10
    B, S, Lg = 2, 10, 30
    M = torch.full((B, S, Lg), 0.0)
    for i in range(6):
        M[0, i, 3 + i] = 6.0
    for i in range(10):
        M[1, i, 5 + i] = 6.0
    seg_mask = torch.zeros(B, S, dtype=torch.bool)
    seg_mask[0, :6] = True
    seg_mask[1, :10] = True
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    end = soft_dp_end_logits(M, seg_mask, germ_mask, *_gaps())
    assert end[0].argmax().item() == 3 + 5         # 8
    assert end[1].argmax().item() == 5 + 9         # 14


def test_module_start_and_end_and_grad():
    torch.manual_seed(0)
    B, S, Lg, d, c0 = 2, 12, 25, 32, 4
    al = SoftDPAligner(d_model=d)
    # identity projections + one-hot (orthogonal) reps -> clean match matrix, isolates
    # the DP/reversal logic from random-rep cosine noise.
    with torch.no_grad():
        al.seg_proj.weight.copy_(torch.eye(d)); al.seg_proj.bias.zero_()
        al.germ_proj.weight.copy_(torch.eye(d)); al.germ_proj.bias.zero_()
    germ = torch.zeros(B, Lg, d)
    for j in range(Lg):
        germ[:, j, j] = 1.0                          # germline position j -> basis vector e_j
    seg = torch.zeros(B, S, d)
    for i in range(S):
        seg[:, i] = germ[:, c0 + i]                  # read i matches germline c0+i
    seg.requires_grad_(True)
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    start, end = al(seg, seg_mask, germ, germ_mask)
    assert start.shape == (B, Lg) and end.shape == (B, Lg)
    # start should be near c0, end near c0+S-1 (allow slack; reps are random but matched)
    assert abs(start[0].argmax().item() - c0) <= 2
    assert abs(end[0].argmax().item() - (c0 + S - 1)) <= 2
    (start.sum() + end.sum()).backward()
    assert torch.isfinite(seg.grad).all()
