"""Retrieval prefilter: top-k shortlist (with forced positive) + batched MaxSim."""
import torch
import torch.nn.functional as F

from alignair.nn.sota.retrieval import retrieve_topk, maxsim_scores_batched, gather_candidates


def test_topk_returns_the_most_similar():
    read = F.normalize(torch.randn(3, 8), dim=-1)
    cand = F.normalize(torch.randn(10, 8), dim=-1)
    idx = retrieve_topk(read, cand, k=4)
    assert idx.shape == (3, 4)
    full = (read @ cand.t()).topk(4, dim=-1).indices
    assert (idx.sort(dim=-1).values == full.sort(dim=-1).values).all()


def test_force_include_guarantees_the_positive_in_the_shortlist():
    torch.manual_seed(0)
    read = F.normalize(torch.randn(5, 8), dim=-1)
    cand = F.normalize(torch.randn(20, 8), dim=-1)
    true = torch.tensor([19, 0, 11, 3, 7])           # arbitrary positives, likely NOT in top-3
    idx = retrieve_topk(read, cand, k=3, force_include=true)
    for b in range(5):
        assert true[b].item() in idx[b].tolist()


def test_force_include_skips_negative_sentinels():
    read = F.normalize(torch.randn(2, 8), dim=-1)
    cand = F.normalize(torch.randn(10, 8), dim=-1)
    true = torch.tensor([-100, 4])                   # row 0 has no gene -> must not be forced
    idx = retrieve_topk(read, cand, k=3, force_include=true)
    assert 4 in idx[1].tolist()
    assert -100 not in idx[0].tolist()               # sentinel never inserted


def test_batched_maxsim_matches_shared_when_all_rows_share_candidates():
    torch.manual_seed(0)
    B, Sq, d, K, Sc = 3, 6, 16, 4, 5
    q = F.normalize(torch.randn(B, Sq, d), dim=-1)
    qm = torch.ones(B, Sq, dtype=torch.bool)
    c = F.normalize(torch.randn(K, Sc, d), dim=-1)
    cm = torch.ones(K, Sc, dtype=torch.bool)
    from alignair.nn.sota.matching import maxsim_scores
    shared = maxsim_scores(q, qm, c, cm)                         # (B, K)
    cb = c[None].expand(B, -1, -1, -1)
    cmb = cm[None].expand(B, -1, -1)
    batched = maxsim_scores_batched(q, qm, cb, cmb)             # (B, K)
    assert torch.allclose(shared, batched, atol=1e-5)


def test_gather_candidates_shapes():
    K, Sc, d, B, k = 10, 5, 8, 3, 4
    tok = torch.randn(K, Sc, d); msk = torch.ones(K, Sc, dtype=torch.bool)
    idx = torch.randint(0, K, (B, k))
    gt, gm = gather_candidates(tok, msk, idx)
    assert gt.shape == (B, k, Sc, d) and gm.shape == (B, k, Sc)
    assert torch.equal(gt[0, 0], tok[idx[0, 0]])
