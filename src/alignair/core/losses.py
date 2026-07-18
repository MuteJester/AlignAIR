"""Faithful hierarchical multi-task loss for AlignAIR (TF ``hierarchical_loss``), with the one
approved deviation: **proper Kendall** task weighting (``loss·exp(-s) + 0.5·s``) via
:class:`alignair.nn.weighting.UncertaintyWeight` — the TF code dropped the counter-term, so its
weights collapsed to uniform.

Segmentation: soft-Gaussian (sigma=1.5) cross-entropy on position logits + auxiliary length/IoU/
hinge regularizers. Classification: multi-label sigmoid BCE with label smoothing 0.1 (+ short-D
penalty). Analysis: mutation/indel MAE, productivity BCE (no smoothing).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AlignAIRConfig
from ..nn.heads.state import STATE_CLASS_WEIGHTS
from ..nn.weighting import UncertaintyWeight

_STATE_IGNORE = -100    # per-position label pad (matches gym_collate's ignore_index)


def genes_of(cfg: AlignAIRConfig) -> list[str]:
    return [s.name for s in cfg.gene_specs]


def make_logvars(cfg: AlignAIRConfig) -> nn.ModuleDict:
    """One UncertaintyWeight per weighted head (owned by the trainer, passed into the loss)."""
    keys = []
    for g in genes_of(cfg):
        keys += [f"{g}_start", f"{g}_end", f"{g}_classification"]
    keys += ["mutation", "indel", "productive", "orientation"]
    if getattr(cfg, "num_chain_types", 1) > 1:                  # multi-chain: chain_type (locus) head
        keys.append("chain_type")
    if getattr(cfg, "state_head", False):                      # per-position edit-state head
        keys.append("state")
    return nn.ModuleDict({k: UncertaintyWeight() for k in keys})


def soft_gaussian_target(gt: torch.Tensor, length: int, sigma: float = 1.5) -> torch.Tensor:
    """Discretized Gaussian over positions ``0..length-1`` centered at ``gt``, renormalized."""
    gt = gt.reshape(-1, 1).clamp(0, length - 1)
    idx = torch.arange(length, device=gt.device, dtype=torch.float32).unsqueeze(0)   # (1, L)
    return F.softmax(-0.5 * (idx - gt) ** 2 / sigma ** 2, dim=-1)                     # (B, L)


def _kendall(uw: UncertaintyWeight, loss: torch.Tensor) -> torch.Tensor:
    return loss * uw() + uw.penalty()


def _soft_ce(logits: torch.Tensor, soft_target: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """Soft-label cross-entropy over positions. ``mask`` (B, L bool) restricts the softmax to the
    valid read positions so the model never competes on padding; the pad terms are dropped from the
    sum (guarding against ``0 * -inf`` NaN)."""
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    logp = F.log_softmax(logits, dim=-1)
    term = soft_target * logp
    if mask is not None:
        term = torch.where(mask, term, term.new_zeros(()))
    return -term.sum(-1).mean()


def hierarchical_loss(out: dict, targets: dict, cfg: AlignAIRConfig, logvars: nn.ModuleDict):
    L = cfg.max_seq_length
    genes = genes_of(cfg)
    parts: dict = {}

    # ---- segmentation: soft-Gaussian CE per boundary (Kendall-weighted) + aux regularizers ----
    pos_mask = out.get("position_mask")                         # restrict CE to the read (not the pad)
    seg = out["v_start"].new_zeros(())
    for g in genes:
        for b in ("start", "end"):
            key = f"{g}_{b}"
            ce = _soft_ce(out[f"{key}_logits"], soft_gaussian_target(targets[key], L), pos_mask)
            seg = seg + _kendall(logvars[key], ce)
    aux_len = out["v_start"].new_zeros(())
    aux_iou = out["v_start"].new_zeros(())
    aux_hinge = out["v_start"].new_zeros(())
    for g in genes:
        sp, ep = out[f"{g}_start"], out[f"{g}_end"]
        st, et = targets[f"{g}_start"].reshape(-1, 1), targets[f"{g}_end"].reshape(-1, 1)
        len_p, len_t = ep - sp, et - st
        aux_len = aux_len + F.huber_loss(len_p, len_t, delta=1.0)
        inter = (torch.minimum(ep, et) - torch.maximum(sp, st)).clamp(min=0)
        union = (len_p.clamp(min=0) + len_t.clamp(min=0) - inter).clamp(min=1e-6)
        aux_iou = aux_iou + (1.0 - (inter / union)).mean()
        aux_hinge = aux_hinge + F.relu(1.0 - len_p).mean()
    seg = seg + 0.1 * aux_len + 0.1 * aux_iou + 0.05 * aux_hinge
    parts["segmentation"] = seg

    # ---- classification: multi-label sigmoid BCE (label smoothing 0.1), Kendall-weighted ----
    clf = out["v_start"].new_zeros(())
    for g in genes:
        y = targets[f"{g}_allele"]
        y_smooth = y * (1.0 - 0.1) + 0.05                       # Keras label_smoothing=0.1
        bce = F.binary_cross_entropy(out[f"{g}_allele"].clamp(1e-7, 1 - 1e-7), y_smooth)
        clf = clf + _kendall(logvars[f"{g}_classification"], bce)
    for spec in cfg.gene_specs:                                 # short-D degenerate-span penalty
        if spec.short_d_penalty:
            d_len = (out[f"{spec.name}_end"] - out[f"{spec.name}_start"]).reshape(-1)
            short_d_prob = out[f"{spec.name}_allele"][:, -1]
            clf = clf + ((d_len < 5).float() * short_d_prob).mean()
    parts["classification"] = clf

    # ---- analysis: mutation/indel MAE, productivity BCE (no smoothing) ----
    mut = F.l1_loss(out["mutation_rate"], targets["mutation_rate"])
    ind = F.l1_loss(out["indel_count"], targets["indel_count"])
    prod = F.binary_cross_entropy(out["productive"].clamp(1e-7, 1 - 1e-7), targets["productive"])
    parts["mutation"] = _kendall(logvars["mutation"], mut)
    parts["indel"] = _kendall(logvars["indel"], ind)
    parts["productive"] = _kendall(logvars["productive"], prod)

    total = parts["segmentation"] + parts["classification"] + parts["mutation"] \
        + parts["indel"] + parts["productive"]

    # ---- per-position edit state: masked, class-weighted CE (germline/sub/insertion/deletion) ----
    if "state_logits" in out and "state_labels" in targets and "state" in logvars:
        sl = out["state_logits"]                                    # (B, L, S)
        w = sl.new_tensor(STATE_CLASS_WEIGHTS)
        st_ce = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), targets["state_labels"].reshape(-1),
                                weight=w, ignore_index=_STATE_IGNORE)
        parts["state"] = _kendall(logvars["state"], st_ce)
        total = total + parts["state"]

    # ---- orientation (supervised only when the batch carries an orientation label) ----
    if "orientation" in targets and "orientation_logits" in out:
        o_loss = F.cross_entropy(out["orientation_logits"], targets["orientation"])
        parts["orientation"] = _kendall(logvars["orientation"], o_loss)
        total = total + parts["orientation"]

    # ---- chain_type / locus (multi-chain only; supervised when the batch carries the label) ----
    if "chain_type" in targets and "chain_type_logits" in out and "chain_type" in logvars:
        ct_loss = F.cross_entropy(out["chain_type_logits"], targets["chain_type"])
        parts["chain_type"] = _kendall(logvars["chain_type"], ct_loss)
        total = total + parts["chain_type"]
    return total, parts
