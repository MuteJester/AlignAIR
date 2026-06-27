from __future__ import annotations
import torch
import torch.nn.functional as F

from ..nn.heads.orientation import apply_orientation
from ..nn.heads.cross_attn_matcher import xattn_match
from ..core.dnalignair import extract_segment
from .reader import build_candidates, reader_set_nce


def _coord_ce(logits, target, Lg):
    # logits (B,Lg) over germline positions; target (B,) true coordinate, clamped into range
    tgt = target.reshape(-1).long().clamp(0, Lg - 1)
    return F.cross_entropy(logits, tgt)


def xattn_losses(model, batch, ref_emb, sib_index, rng, n_sib: int = 6, n_rand: int = 6):
    """Multi-task supervised loss for XAttnAligner on a GenAIRR gym batch: orientation CE +
    region CE + allele set-NCE + germline-span CE. Teacher-forced (true orientation + true region
    labels). Returns (total, {orientation, region, allele, gstart, gend})."""
    tokens, mask = batch["tokens"], batch["mask"]
    ori_logits = model.orientation_head(tokens, mask)
    L_ori = F.cross_entropy(ori_logits, batch["orientation_id"].reshape(-1).long())

    canon = apply_orientation(tokens, mask, batch["orientation_id"].reshape(-1).long())
    reps = model.backbone.forward_positions(canon, mask)
    rdec = model.region_tagger(reps, mask)
    rl = rdec["region_logits"]                                                # (B,L,R)
    B, L, R = rl.shape
    reg_ce = F.cross_entropy(rl.reshape(B * L, R), batch["region_labels"].reshape(B * L).long(),
                             reduction="none").reshape(B, L)
    L_reg = (reg_ce * mask.float()).sum() / mask.float().sum().clamp(min=1)

    genes = ["v", "j"] + (["d"] if "d_allele" in batch else [])
    L_allele = reps.new_zeros(())
    L_gs = reps.new_zeros(())
    L_ge = reps.new_zeros(())
    for g in genes:
        G = g.upper()
        emb = ref_emb[G]
        seg, seg_mask = extract_segment(reps, mask, batch["region_labels"].long(), G)
        cand_idx, pos_mask = build_candidates(batch[f"{g}_primary_idx"], batch[f"{g}_allele"],
                                              sib_index[G], rng, n_sib=n_sib, n_rand=n_rand)
        out = xattn_match(model.matcher, seg, seg_mask, emb["pos_reps"], emb["pos_mask"], cand_idx)
        Lg = emb["pos_reps"].shape[1]
        L_allele = L_allele + reader_set_nce(out["allele_logits"], pos_mask)
        # germline_start is the 0-based inclusive first matched germline position; germline_end is
        # GenAIRR's exclusive (one-past) end, so the last-token pointer target is germline_end - 1.
        L_gs = L_gs + _coord_ce(out["gstart_logits"][:, 0], batch[f"{g}_germline_start"], Lg)
        L_ge = L_ge + _coord_ce(out["gend_logits"][:, 0], batch[f"{g}_germline_end"] - 1, Lg)

    total = L_ori + L_reg + L_allele + 0.5 * (L_gs + L_ge)
    return total, {"orientation": L_ori.detach(), "region": L_reg.detach(),
                   "allele": L_allele.detach(), "gstart": L_gs.detach(), "gend": L_ge.detach()}
