"""Teacher-forced germline-coordinate logits for training.

For each gene, re-encode the true segment's tokens with the (shared) germline
encoder into per-position reps and align them to the true allele's per-position
germline reps (same encoder space) via the model's germline aligner. When
``state_logits`` is supplied, the per-position SHM reliability of the segment is
gathered and passed through so the pointer aligner's base-match channel can
down-weight likely-mutated positions."""
import torch

from ..core.dnalignair import extract_segment_tokens, extract_segment


def compute_germline_logits(model, tokens, mask, batch, ref_emb, has_d: bool,
                            region_labels=None, allele_idx: dict | None = None,
                            state_logits=None, reps=None, return_band=False):
    """Per-gene germline (start, end) logits.

    Teacher-forced (default): segment is extracted with the TRUE region labels and
    aligned to the coordinate-bearing primary allele's reps. Pass ``region_labels``
    (e.g. predicted) and/or ``allele_idx`` ({gene_upper: (B,) idx}, e.g. predicted
    top-1) to measure the true end-to-end pipeline instead. Pass ``state_logits``
    (B, L, n_states) to enable SHM reliability gating in the pointer aligner."""
    genes = ["v", "j"] + (["d"] if has_d else [])
    rl = region_labels if region_labels is not None else batch["region_labels"]
    out = {}
    band = {}
    for g in genes:
        G = g.upper()
        seg_tok, seg_mask = extract_segment_tokens(tokens, mask, rl, G)
        if getattr(model, "seed_extend", False):
            # ONE shared encoder: read the segment reps OFF the backbone reps (no re-encode) when
            # available, else re-encode the segment through the SHARED encoder (READ). Never the
            # separate GermlineEncoder (it does not exist on the seed_extend path).
            if reps is not None:
                seg_reps, _ = extract_segment(reps, mask, rl, G)              # (B, S, d), no re-encode
            else:
                from ..nn.encoder.shared import SharedNucleotideEncoder
                seg_reps = model.backbone.forward_positions(seg_tok, seg_mask,
                                                            token_type=SharedNucleotideEncoder.READ)
        else:
            seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)  # (B, S, d)
        if allele_idx is not None and G in allele_idx:
            idx = allele_idx[G]
        elif f"{g}_primary_idx" in batch:
            idx = batch[f"{g}_primary_idx"]
        else:
            multihot = batch[f"{g}_allele"]
            idx = torch.where(multihot.sum(dim=1) > 0, multihot.argmax(dim=1),
                              torch.zeros(multihot.shape[0], dtype=torch.long,
                                          device=multihot.device))
        germ_reps = ref_emb[G]["pos_reps"][idx]       # (B, Lg, d)
        germ_mask = ref_emb[G]["pos_mask"][idx]       # (B, Lg)
        germ_tok = ref_emb[G]["pos_tok"][idx]         # (B, Lg) for base-match
        seg_rel = None
        if state_logits is not None:
            from ..nn.heads.state import state_reliability
            seg_state, _ = extract_segment(state_logits, mask, rl, G)
            seg_rel = state_reliability(seg_state)
        if getattr(model, "seed_extend", False):
            # compute the band logits ONCE: they place the DP band (argmax center) AND are
            # supervised by band_offset_loss (the band head's only gradient — center is argmax).
            bl = model.band_logits(seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok)
            band[g] = bl
            center = bl.argmax(dim=-1)
            w = getattr(model.config, "band_width", 16)
            out[g] = model.aligner(seg_reps, seg_mask, germ_reps, germ_mask, center, w,
                                   seg_tok=seg_tok, germ_tok=germ_tok, seg_reliability=seg_rel)
        else:
            out[g] = model.germline_coords(seg_reps, seg_mask, germ_reps, germ_mask,
                                           seg_tok=seg_tok, germ_tok=germ_tok,
                                           seg_reliability=seg_rel)
    if return_band:
        return out, band
    return out
