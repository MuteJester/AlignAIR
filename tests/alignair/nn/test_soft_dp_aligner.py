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


def test_alignment_score_ranks_true_germline_above_mismatched():
    # the matching germline should score higher than a mismatched one (allele-reader primitive)
    import torch
    from alignair.nn.soft_dp_aligner import SoftDPAligner
    torch.manual_seed(0)
    B, S, Lg, d, c0 = 1, 12, 25, 32, 4
    al = SoftDPAligner(d_model=d)
    with torch.no_grad():
        al.seg_proj.weight.copy_(torch.eye(d)); al.seg_proj.bias.zero_()
        al.germ_proj.weight.copy_(torch.eye(d)); al.germ_proj.bias.zero_()
    germ_true = torch.zeros(B, Lg, d)
    for j in range(Lg):
        germ_true[:, j, j] = 1.0
    seg = torch.stack([germ_true[:, c0 + i] for i in range(S)], dim=1).squeeze(2) \
        if False else torch.zeros(B, S, d)
    for i in range(S):
        seg[:, i] = germ_true[:, c0 + i]
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    # genuinely mismatched germline: the segment's matching content is absent
    # (zero out the region it would align to; a mere shift just relabels positions)
    germ_wrong = germ_true.clone()
    germ_wrong[:, c0:c0 + S] = 0.0
    s_true = al.alignment_score(seg, seg_mask, germ_true, germ_mask)
    s_wrong = al.alignment_score(seg, seg_mask, germ_wrong, germ_mask)
    assert s_true.item() > s_wrong.item()


def test_state_reliability_downweights_shm_mismatch():
    # Heavy-SHM robustness: a mismatch at a position flagged 'substitution' (low reliability)
    # should cost FAR less alignment score than the same mismatch at a 'germline' position.
    torch.manual_seed(0)
    B, S, Lg, d = 1, 10, 10, 16
    al = SoftDPAligner(d_model=d, match_floor=1.0)
    reps = torch.randn(B, Lg, d)
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    seg_tok = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2]])
    germ = seg_tok.clone(); germ[0, 5] = (germ[0, 5] % 4) + 1          # one mismatch at pos 5
    rel_high = torch.ones(B, S)                                        # all 'germline' (reliable)
    rel_low = torch.ones(B, S); rel_low[0, 5] = 0.0                    # pos 5 'substitution'
    s_penalized = al.alignment_score(reps, seg_mask, reps, germ_mask,
                                     seg_tok=seg_tok, germ_tok=germ, seg_reliability=rel_high)
    s_forgiven = al.alignment_score(reps, seg_mask, reps, germ_mask,
                                    seg_tok=seg_tok, germ_tok=germ, seg_reliability=rel_low)
    assert s_forgiven.item() > s_penalized.item()                     # SHM mismatch forgiven


def test_state_reliability_helper_low_at_substitution():
    from alignair.nn.state_head import state_reliability, STATE_INDEX
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, STATE_INDEX["germline"]] = 10.0                       # confident germline
    logits[0, 1, STATE_INDEX["substitution"]] = 10.0                  # confident substitution
    r = state_reliability(logits, r_min=0.25)
    assert r[0, 0] > 0.9                                              # reliable
    assert abs(r[0, 1].item() - 0.25) < 0.05                          # floored at r_min


def test_match_floor_keeps_raw_token_channel_load_bearing():
    # Novel-allele guarantee: even if training drives the LEARNED base-match weight to
    # zero, the FLOORED raw-token channel must still rank an exact-base germline above a
    # SNP-differing sibling — using only ACGT tokens (reps identical, so cosine is a tie).
    torch.manual_seed(0)
    B, S, Lg, d = 1, 12, 12, 16
    al = SoftDPAligner(d_model=d, match_floor=1.0)
    with torch.no_grad():
        al._match_weight.fill_(-50.0)        # softplus(-50) ~ 0 -> learned bonus gone
    reps = torch.randn(B, Lg, d)             # identical reps for seg & both germlines (cosine tie)
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    seg_tok = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])   # ACGT pattern (1..4)
    germ_true = seg_tok.clone()
    germ_snp = seg_tok.clone(); germ_snp[0, 6] = 1 if germ_snp[0, 6] != 1 else 2  # one SNP
    s_true = al.alignment_score(reps, seg_mask, reps, germ_mask,
                                seg_tok=seg_tok, germ_tok=germ_true)
    s_snp = al.alignment_score(reps, seg_mask, reps, germ_mask,
                               seg_tok=seg_tok, germ_tok=germ_snp)
    assert s_true.item() > s_snp.item()      # floored raw-token channel resolves the SNP
