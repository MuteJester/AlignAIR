import random
import torch

from alignair.nn.soft_dp_aligner import SoftDPAligner
from alignair.training.reader import (build_candidates, reader_scores, reader_set_nce,
                                      perturb_germline_tokens, reader_novel_positive)


class _Gene:
    def __init__(self, names):
        self.names = names


class _RS:
    def __init__(self):
        # gene A: a*01,a*02 (siblings); gene B: b*01; gene C: c*01
        self.genes = {"V": _Gene(["IGHVA*01", "IGHVA*02", "IGHVB*01", "IGHVC*01"])}

    def gene(self, g):
        return self.genes[g.upper()]


def test_build_candidates_includes_primary_and_siblings():
    from alignair.training.reader import build_sibling_index
    sib = build_sibling_index(_RS())["V"]
    primary = torch.tensor([0, 2])                       # a*01, b*01
    multihot = torch.zeros(2, 4); multihot[0, 0] = 1; multihot[1, 2] = 1
    cand, pos = build_candidates(primary, multihot, sib, random.Random(0), n_sib=2, n_rand=1)
    assert cand.shape == (2, 4) and pos.shape == (2, 4)
    assert cand[0, 0].item() == 0 and pos[0, 0].item() == 1.0    # primary positive
    assert 1 in cand[0].tolist()                                  # sibling a*02 included for a*01


def test_reader_nce_trains_alignment_score_to_discriminate():
    # toy: segment matches candidate 0's germline; NCE should push its score highest
    torch.manual_seed(0)
    B, C, S, Lg, d = 4, 5, 10, 16, 16
    al = SoftDPAligner(d_model=d)
    with torch.no_grad():
        al.seg_proj.weight.copy_(torch.eye(d)); al.seg_proj.bias.zero_()
        al.germ_proj.weight.copy_(torch.eye(d)); al.germ_proj.bias.zero_()
    # K germlines = one-hot rows; candidate 0 is the true one matching the segment
    K = 8
    pos_reps = torch.zeros(K, Lg, d)
    for k in range(K):
        for j in range(Lg):
            pos_reps[k, j, (j + k) % d] = 1.0
    pos_mask_ref = torch.ones(K, Lg, dtype=torch.bool)
    cand_idx = torch.stack([torch.tensor([0, 1, 2, 3, 4]) for _ in range(B)])
    pos_mask = torch.zeros(B, C); pos_mask[:, 0] = 1.0
    seg = pos_reps[0, :S].unsqueeze(0).expand(B, S, d).clone()   # matches candidate 0
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    opt = torch.optim.Adam(al.parameters(), lr=0.05)
    first = last = None
    for step in range(60):
        scores = reader_scores(al, seg, seg_mask, cand_idx, pos_reps, pos_mask_ref)
        loss = reader_set_nce(scores, pos_mask)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0: first = loss.item()
        last = loss.item()
    assert last < first
    scores = reader_scores(al, seg, seg_mask, cand_idx, pos_reps, pos_mask_ref)
    assert scores.argmax(-1).float().mean().item() == 0.0       # true candidate ranked top


def test_perturb_germline_tokens_changes_exactly_k_bases():
    gen = torch.Generator().manual_seed(0)
    tok = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 0, 0]])         # last two are pad (0)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    out = perturb_germline_tokens(tok, mask, k=3, gen=gen)
    diff = (out != tok)
    assert int(diff.sum()) == 3                                  # exactly k substitutions
    assert not diff[0, 8:].any()                                 # padding untouched (stays 0)
    assert ((out[mask] >= 1) & (out[mask] <= 4)).all()           # valid bases stay valid bases


def test_reader_novel_positive_ranks_perturbed_true_over_random_sibling():
    # the SNP-perturbed TRUE germline (novel stand-in) should still out-align a random
    # unrelated sibling — proving the raw-token channel carries unseen germlines.
    torch.manual_seed(0)
    B, S, Lg, d = 3, 12, 14, 16
    al = SoftDPAligner(d_model=d, match_floor=1.0)
    with torch.no_grad():
        al.seg_proj.weight.copy_(torch.eye(d)); al.seg_proj.bias.zero_()
        al.germ_proj.weight.copy_(torch.eye(d)); al.germ_proj.bias.zero_()

    class _Enc:  # identity position encoder (reps tie, so only tokens discriminate)
        def forward_positions(self, tok, mask):
            return torch.randn(tok.shape[0], tok.shape[1], d)

    pos_tok = torch.randint(1, 5, (B, Lg))
    pos_mask = torch.ones(B, Lg, dtype=torch.bool)
    seg_tok = pos_tok[:, :S].clone()                              # read = first S germline bases
    seg_reps = torch.randn(B, S, d)
    seg_mask = torch.ones(B, S, dtype=torch.bool)
    prim = torch.arange(B)
    gen = torch.Generator().manual_seed(1)
    s_novel = reader_novel_positive(al, _Enc(), seg_reps, seg_mask, seg_tok, prim,
                                    pos_tok, pos_mask, k=1, gen=gen)
    # an unrelated germline (shuffled rows) as the negative
    wrong = pos_tok[torch.tensor([1, 2, 0])]
    s_wrong = al.alignment_score(seg_reps, seg_mask, torch.randn(B, Lg, d), pos_mask,
                                 seg_tok=seg_tok, germ_tok=wrong)
    assert (s_novel > s_wrong).float().mean() >= 2 / 3           # perturbed-true wins in general
