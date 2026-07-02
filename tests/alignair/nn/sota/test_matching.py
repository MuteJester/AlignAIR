"""Token-level (MaxSim) matching — the ceiling-breaker."""
import torch
import torch.nn.functional as F

from alignair.nn.sota.matching import maxsim_scores, TokenMatch, contrastive_match_loss


def _norm(x):
    return F.normalize(x, dim=-1)


def test_maxsim_shapes_and_masking():
    B, Sq, K, Sc, d = 3, 7, 5, 9, 16
    q = _norm(torch.randn(B, Sq, d)); qm = torch.ones(B, Sq, dtype=torch.bool)
    c = _norm(torch.randn(K, Sc, d)); cm = torch.ones(K, Sc, dtype=torch.bool)
    s = maxsim_scores(q, qm, c, cm)
    assert s.shape == (B, K)
    # padding a candidate's real tokens away must not raise the score
    cm2 = cm.clone(); cm2[0, 3:] = False
    assert torch.isfinite(maxsim_scores(q, qm, c, cm2)).all()


def test_maxsim_ranks_the_exact_germline_first():
    """MaxSim is a cheap RETRIEVAL prefilter, not the fine discriminator: it must rank the exact
    germline above a differing one. (Fine sibling discrimination is the LEARNED fusion's job —
    a fixed similarity provably washes out a single SNP over a long sequence; see the design spec
    and the falsified 'MaxSim beats pooled' experiment.)"""
    torch.manual_seed(0)
    S, d = 12, 32
    read = _norm(torch.randn(1, S, d))
    true = read.clone()
    other = read.clone(); other[0, 5] = _norm(torch.randn(1, d))
    m = torch.ones(1, S, dtype=torch.bool)
    mx = maxsim_scores(read, m, torch.cat([true, other]), torch.ones(2, S, dtype=torch.bool))[0]
    assert mx[0] > mx[1]                              # exact germline ranked first


def test_token_match_temperature_and_genotype_mask():
    B, Sq, K, Sc, d = 2, 6, 4, 8, 16
    q = _norm(torch.randn(B, Sq, d)); qm = torch.ones(B, Sq, dtype=torch.bool)
    c = _norm(torch.randn(K, Sc, d)); cm = torch.ones(K, Sc, dtype=torch.bool)
    head = TokenMatch()
    logits = head(q, qm, c, cm)
    assert logits.shape == (B, K)
    gmask = torch.tensor([True, False, True, True])  # allele 1 not in this genotype
    masked = head(q, qm, c, cm, candidate_mask=gmask)
    assert torch.isinf(masked[:, 1]).all() and (masked[:, 1] < 0).all()


def test_contrastive_loss_trains_toward_the_positive():
    torch.manual_seed(0)
    B, Sq, K, Sc, d = 8, 6, 6, 6, 16
    q = _norm(torch.randn(B, Sq, d))
    c = _norm(torch.randn(K, Sc, d)).requires_grad_(False)
    qm = torch.ones(B, Sq, dtype=torch.bool); cm = torch.ones(K, Sc, dtype=torch.bool)
    target = torch.zeros(B, K); target[torch.arange(B), torch.arange(B) % K] = 1.0
    head = TokenMatch()
    qp = torch.nn.Parameter(q.clone())
    opt = torch.optim.Adam([qp, *head.parameters()], lr=5e-2)
    l0 = contrastive_match_loss(head(qp, qm, c, cm), target).item()
    for _ in range(100):
        opt.zero_grad()
        loss = contrastive_match_loss(head(F.normalize(qp, dim=-1), qm, c, cm), target)
        loss.backward(); opt.step()
    assert loss.item() < 0.5 * l0
