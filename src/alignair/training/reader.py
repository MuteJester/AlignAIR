"""Allele-reader training: teach the seed-and-extend banded DP to DISCRIMINATE alleles
(not just predict coordinates).

The untrained alignment_score ranks alleles no better than chance because it was only
supervised for germline coordinates against the true allele. Here we train it
contrastively: for each observed segment, score a candidate set = true allele(s) +
SIBLING hard-negatives (same gene, 1-2 SNP apart — the cases that matter) + random
negatives, with a multi-positive set-NCE loss. Scoring is the exact banded log-partition
(SeedExtendAligner.alignment_score); set-NCE on GT labels is the core signal.
"""
from collections import defaultdict

import torch

NEG = -1e4


def build_sibling_index(reference_set) -> dict:
    """Per gene: (gene_name_of_allele list, {gene_name: [allele_idx,...]})."""
    out = {}
    for G in [g for g in ("V", "D", "J") if g in reference_set.genes]:
        names = reference_set.gene(G).names
        gene_of = [nm.split("*")[0] for nm in names]
        by_gene = defaultdict(list)
        for i, gn in enumerate(gene_of):
            by_gene[gn].append(i)
        out[G] = (gene_of, dict(by_gene))
    return out


def build_candidates(primary_idx, multihot, sib_index_G, rng,
                     n_sib: int = 6, n_rand: int = 6) -> tuple:
    """Per example: candidate indices [true primary + siblings + randoms] and a
    positive mask (1 where the candidate is in the example's true allele set).
    Returns cand_idx (B,C) long, pos_mask (B,C) float. C = 1 + n_sib + n_rand."""
    gene_of, by_gene = sib_index_G
    B, K = multihot.shape
    C = 1 + n_sib + n_rand
    device = primary_idx.device
    cand = torch.zeros(B, C, dtype=torch.long, device=device)
    for i in range(B):
        prim = int(primary_idx[i])
        is_valid = prim >= 0
        prim_clamped = prim if is_valid else 0
        cand[i, 0] = prim_clamped
        sibs = [j for j in by_gene.get(gene_of[prim_clamped], []) if j != prim_clamped]
        rng.shuffle(sibs)
        sib_sel = sibs[:n_sib]
        pool = list(range(K))
        rng.shuffle(pool)
        rand_sel = [j for j in pool if j != prim_clamped and j not in sib_sel][:n_rand]
        fill = (sib_sel + rand_sel + [prim_clamped] * C)[:C - 1]      # pad with prim if short
        cand[i, 1:] = torch.tensor(fill, dtype=torch.long, device=device)
    pos_mask = multihot.gather(1, cand)                       # (B,C) 1 where candidate is a true allele
    pos_mask[:, 0] = 1.0                                       # primary is always positive
    valid_mask = (primary_idx >= 0).float().unsqueeze(1)
    pos_mask = pos_mask * valid_mask
    return cand, pos_mask


def reader_scores_banded(model, seg_reps, seg_mask, cand_idx, pos_reps, pos_mask_ref,
                         pos_tok_ref, seg_tok, seg_reliability=None, w: int = 16) -> torch.Tensor:
    """seed_extend reader scores (B,C): for each candidate the band head places a band
    (DETACHED argmax center) and the EXACT banded SeedExtendAligner.alignment_score
    (log-partition, rule 1) discriminates. The band center is detached so the reader loss
    trains the DP reader/emissions/encoder while the band head keeps its own offset-CE
    supervision. Base-match + SHM reliability are preserved in the score path."""
    B, C = cand_idx.shape
    S, d = seg_reps.shape[1], seg_reps.shape[2]
    seg = seg_reps.unsqueeze(1).expand(B, C, S, d).reshape(B * C, S, d)
    sm = seg_mask.unsqueeze(1).expand(B, C, S).reshape(B * C, S)
    flat = cand_idx.reshape(-1)
    germ = pos_reps[flat]; gm = pos_mask_ref[flat]; gt = pos_tok_ref[flat]
    st = seg_tok.unsqueeze(1).expand(B, C, S).reshape(B * C, S)
    rel = (seg_reliability.unsqueeze(1).expand(B, C, S).reshape(B * C, S)
           if seg_reliability is not None else None)
    with torch.no_grad():
        center = model.band_head(seg, sm, germ, gm, st, gt).argmax(dim=-1)
    sc = model.aligner.alignment_score(seg, sm, germ, gm, center, w,
                                       seg_tok=st, germ_tok=gt, seg_reliability=rel)
    return sc.reshape(B, C)


def reader_set_nce(scores: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """Multi-positive set-NCE: -log( sum_pos e^s / sum_all e^s ). Ranks the true
    allele set above the (sibling + random) negatives."""
    valid = (pos_mask.sum(dim=-1) > 0)
    pos_scores = scores.masked_fill(pos_mask <= 0, NEG)
    loss = torch.logsumexp(scores, dim=-1) - torch.logsumexp(pos_scores, dim=-1)
    if not valid.any():
        return scores.new_zeros(())
    return loss[valid].mean()
