"""Teacher-forced germline-coordinate logits for training.

For each gene, gather the true segment reps (from the GT region labels) and the
true allele's per-position germline reps (first positive in the multi-hot label),
then run the model's germline aligner."""
import torch

from ..core.dnalignair import extract_segment


def compute_germline_logits(model, reps, mask, batch, ref_emb, has_d: bool):
    genes = ["v", "j"] + (["d"] if has_d else [])
    out = {}
    for g in genes:
        seg, seg_mask = extract_segment(reps, mask, batch["region_labels"], g.upper())
        multihot = batch[f"{g}_allele"]                       # (B, K)
        has_pos = multihot.sum(dim=1) > 0
        idx = multihot.argmax(dim=1)                          # (B,)
        idx = torch.where(has_pos, idx, torch.zeros_like(idx))
        germ_reps = ref_emb[g.upper()]["pos_reps"][idx]       # (B, Lg, d)
        germ_mask = ref_emb[g.upper()]["pos_mask"][idx]       # (B, Lg)
        out[g] = model.germline_coords(seg, seg_mask, germ_reps, germ_mask)
    return out
