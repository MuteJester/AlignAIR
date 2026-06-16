"""Teacher-forced germline-coordinate logits for training.

For each gene, re-encode the true segment's tokens with the (shared) germline
encoder into per-position reps and align them to the true allele's per-position
germline reps (same encoder space) via the model's germline aligner."""
import torch

from ..core.dnalignair import extract_segment_tokens


def compute_germline_logits(model, tokens, mask, batch, ref_emb, has_d: bool):
    genes = ["v", "j"] + (["d"] if has_d else [])
    out = {}
    for g in genes:
        seg_tok, seg_mask = extract_segment_tokens(tokens, mask, batch["region_labels"], g.upper())
        seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)  # (B, S, d)
        multihot = batch[f"{g}_allele"]                       # (B, K)
        has_pos = multihot.sum(dim=1) > 0
        idx = multihot.argmax(dim=1)
        idx = torch.where(has_pos, idx, torch.zeros_like(idx))
        germ_reps = ref_emb[g.upper()]["pos_reps"][idx]       # (B, Lg, d)
        germ_mask = ref_emb[g.upper()]["pos_mask"][idx]       # (B, Lg)
        out[g] = model.germline_coords(seg_reps, seg_mask, germ_reps, germ_mask)
    return out
