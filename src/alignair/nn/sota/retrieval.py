"""Retrieval prefilter for a large reference.

A real germline reference is hundreds of alleles per gene — too many for token-level MaxSim over
all of them at once. So we do the standard retrieve-then-rerank: score every allele cheaply with a
pooled cosine similarity, keep the top-k, and run the expensive fusion + MaxSim discrimination
only on that shortlist. At training time the true allele is force-included so the contrastive loss
always sees its positive; the rest of the shortlist are the (hard) negatives.

`maxsim_scores_batched` is the per-batch-item version of `matching.maxsim_scores` — after top-k the
candidate set differs per read, so candidates carry a batch dimension.
"""
import torch


def retrieve_topk(read_pooled: torch.Tensor, cand_pooled: torch.Tensor, k: int,
                  force_include: torch.Tensor | None = None) -> torch.Tensor:
    """read_pooled (B, d) and cand_pooled (K, d), both L2-normalized. -> top-k indices (B, k).

    force_include (B,) optional: an allele index (>=0) to guarantee in each row's shortlist
    (the training positive); negative entries (absent gene / ignore) are skipped."""
    k = min(k, cand_pooled.shape[0])
    sims = read_pooled @ cand_pooled.t()                       # (B, K)
    idx = sims.topk(k, dim=-1).indices                         # (B, k)
    if force_include is not None:
        valid = force_include >= 0
        present = (idx == force_include[:, None]).any(dim=-1)  # already in shortlist?
        need = valid & ~present
        idx[need, -1] = force_include[need]                    # drop the weakest, insert the positive
    return idx


def maxsim_scores_batched(q: torch.Tensor, q_mask: torch.Tensor,
                          c: torch.Tensor, c_mask: torch.Tensor) -> torch.Tensor:
    """Late-interaction similarity with a per-batch candidate set.

    q (B, Sq, d), q_mask (B, Sq); c (B, K, Sc, d), c_mask (B, K, Sc). All L2-normalized on d.
    -> (B, K) = mean over valid query tokens of ( max over valid candidate tokens cos )."""
    sim = torch.einsum("bqd,bkcd->bkqc", q, c)                 # (B, K, Sq, Sc)
    sim = sim.masked_fill(~c_mask[:, :, None, :], float("-inf"))
    maxc = sim.max(dim=-1).values                              # (B, K, Sq)
    qm = q_mask[:, None, :].to(maxc.dtype)
    maxc = maxc.masked_fill(qm == 0, 0.0)
    return (maxc * qm).sum(dim=-1) / qm.sum(dim=-1).clamp(min=1.0)


def gather_candidates(cand_tokens: torch.Tensor, cand_mask: torch.Tensor,
                      idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a per-batch shortlist from a shared candidate bank.

    cand_tokens (K, Sc, d), cand_mask (K, Sc), idx (B, k) -> ((B, k, Sc, d), (B, k, Sc))."""
    tok = cand_tokens[idx]                                     # (B, k, Sc, d)
    msk = cand_mask[idx]                                       # (B, k, Sc)
    return tok, msk
