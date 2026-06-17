"""Teacher-forced germline-coordinate logits for training.

For each gene, re-encode the true segment's tokens with the (shared) germline
encoder into per-position reps and align them to the true allele's per-position
germline reps (same encoder space) via the model's germline aligner."""
import torch

from ..core.dnalignair import extract_segment_tokens


def compute_germline_logits(model, tokens, mask, batch, ref_emb, has_d: bool,
                            region_labels=None, allele_idx: dict | None = None):
    """Per-gene germline (start, end) logits.

    Teacher-forced (default): segment is extracted with the TRUE region labels and
    aligned to the coordinate-bearing primary allele's reps. Pass ``region_labels``
    (e.g. predicted) and/or ``allele_idx`` ({gene_upper: (B,) idx}, e.g. predicted
    top-1) to measure the true end-to-end pipeline instead."""
    genes = ["v", "j"] + (["d"] if has_d else [])
    rl = region_labels if region_labels is not None else batch["region_labels"]
    out = {}
    for g in genes:
        seg_tok, seg_mask = extract_segment_tokens(tokens, mask, rl, g.upper())
        seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)  # (B, S, d)
        if allele_idx is not None and g.upper() in allele_idx:
            idx = allele_idx[g.upper()]
        elif f"{g}_primary_idx" in batch:
            idx = batch[f"{g}_primary_idx"]
        else:
            multihot = batch[f"{g}_allele"]
            idx = torch.where(multihot.sum(dim=1) > 0, multihot.argmax(dim=1),
                              torch.zeros(multihot.shape[0], dtype=torch.long, device=multihot.device))
        germ_reps = ref_emb[g.upper()]["pos_reps"][idx]       # (B, Lg, d)
        germ_mask = ref_emb[g.upper()]["pos_mask"][idx]       # (B, Lg)
        out[g] = model.germline_coords(seg_reps, seg_mask, germ_reps, germ_mask)
    return out
