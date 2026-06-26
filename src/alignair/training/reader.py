"""Allele-reader training: teach the differentiable soft-DP aligner to DISCRIMINATE
alleles (not just predict coordinates).

The untrained alignment_score ranks alleles no better than chance because it was only
supervised for germline coordinates against the true allele. Here we train it
contrastively: for each observed segment, score a candidate set = true allele(s) +
SIBLING hard-negatives (same gene, 1-2 SNP apart — the cases that matter) + random
negatives, with a multi-positive set-NCE loss. SW can later be added as a distillation
teacher; set-NCE on GT labels is the core signal.
"""
from collections import defaultdict

import torch
import torch.nn.functional as F

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
        cand[i, 0] = prim
        sibs = [j for j in by_gene.get(gene_of[prim], []) if j != prim]
        rng.shuffle(sibs)
        sib_sel = sibs[:n_sib]
        pool = list(range(K))
        rng.shuffle(pool)
        rand_sel = [j for j in pool if j != prim and j not in sib_sel][:n_rand]
        fill = (sib_sel + rand_sel + [prim] * C)[:C - 1]      # pad with prim if short
        cand[i, 1:] = torch.tensor(fill, dtype=torch.long, device=device)
    pos_mask = multihot.gather(1, cand)                       # (B,C) 1 where candidate is a true allele
    pos_mask[:, 0] = 1.0                                       # primary is always positive
    return cand, pos_mask


def reader_scores(aligner, seg_reps, seg_mask, cand_idx, pos_reps, pos_mask_ref,
                  seg_tok=None, germ_tok_ref=None, seg_reliability=None) -> torch.Tensor:
    """Align the observed segment against each candidate germline -> (B,C) scores.
    Pass seg_tok (B,S) and germ_tok_ref (K,Lg) to enable the base-match channel; pass
    seg_reliability (B,S) to down-weight that channel at likely-SHM positions."""
    B, C = cand_idx.shape
    S, d = seg_reps.shape[1], seg_reps.shape[2]
    seg = seg_reps.unsqueeze(1).expand(B, C, S, d).reshape(B * C, S, d)
    sm = seg_mask.unsqueeze(1).expand(B, C, S).reshape(B * C, S)
    flat = cand_idx.reshape(-1)
    germ = pos_reps[flat]                                     # (B*C, Lg, d)
    gm = pos_mask_ref[flat]                                   # (B*C, Lg)
    st = seg_tok.unsqueeze(1).expand(B, C, S).reshape(B * C, S) if seg_tok is not None else None
    gt = germ_tok_ref[flat] if germ_tok_ref is not None else None
    rel = (seg_reliability.unsqueeze(1).expand(B, C, S).reshape(B * C, S)
           if seg_reliability is not None else None)
    return aligner.alignment_score(seg, sm, germ, gm, seg_tok=st, germ_tok=gt,
                                   seg_reliability=rel).reshape(B, C)


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
    pos_scores = scores.masked_fill(pos_mask <= 0, NEG)
    return (torch.logsumexp(scores, dim=-1) - torch.logsumexp(pos_scores, dim=-1)).mean()


def perturb_germline_tokens(tok: torch.Tensor, mask: torch.Tensor, k: int,
                            gen: torch.Generator) -> torch.Tensor:
    """Substitute k random valid (ACGT = token ids 1..4) positions per row with a
    DIFFERENT base — a synthetic NOVEL allele a few SNPs from the real germline. Used to
    train the reader to align observed reads to germlines it has NOT embedded into weights
    (the dynamic-reference property): the floored raw-token channel must carry the match."""
    out = tok.clone()
    B, Lg = tok.shape
    valid = mask & (tok >= 1) & (tok <= 4)                       # (B,Lg) substitutable
    for b in range(B):
        idx = valid[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        sel = idx[torch.randperm(idx.numel(), generator=gen, device=tok.device)[:k]]
        # shift each selected base by 1..3 (mod 4) within {1,2,3,4} -> guaranteed change
        shift = torch.randint(1, 4, (sel.numel(),), generator=gen, device=tok.device)
        out[b, sel] = ((tok[b, sel] - 1 + shift) % 4) + 1
    return out


def reader_novel_positive(aligner, germline_encoder, seg_reps, seg_mask, seg_tok,
                          prim_idx, pos_tok_ref, pos_mask_ref, k, gen,
                          seg_reliability=None) -> torch.Tensor:
    """Align each example's observed segment against a SNP-perturbed copy of its OWN true
    germline (re-encoded fresh, so its embedding was never optimised) -> (B,) score. This
    is the novel-allele positive that replaces column 0 of the candidate scores."""
    g_tok = perturb_germline_tokens(pos_tok_ref[prim_idx], pos_mask_ref[prim_idx], k, gen)
    g_mask = pos_mask_ref[prim_idx]
    g_reps = germline_encoder.forward_positions(g_tok, g_mask)   # (B, Lg, d) re-encoded
    return aligner.alignment_score(seg_reps, seg_mask, g_reps, g_mask,
                                   seg_tok=seg_tok, germ_tok=g_tok,
                                   seg_reliability=seg_reliability)
