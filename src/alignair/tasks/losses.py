"""Alignment objectives selected by score semantics, independent of sequence alphabet."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..core.anchors import ANCHOR_GENES, ANCHOR_KEYS, ANCHOR_LOGITS, codon_starts
from ..core.d_presence import D_PRESENCE, D_PRESENCE_LOGITS
from ..core.config import AlignAIRConfig
from ..neural.heads.state import STATE_CLASS_WEIGHTS
from ..neural.weighting import UncertaintyWeight
from .supervision import supervised_rows, categorical_rows, productive_rows, target_rows
from .functional import (_kendall, _mean_rows, _soft_ce, soft_gaussian_target,
                         soft_cross_entropy_rows as _soft_ce_per_sample)

_IGNORE = -100                     # the padding of the per-position labels (PositionLabels): a position no term reads
ANCHOR_SIGMA = 1.0                 # the spread of an observed anchor's soft target (``AlignAIRConfig.anchor_heads``)


def genes_of(cfg: AlignAIRConfig) -> list[str]:
    return [s.name for s in cfg.gene_specs]


def loss_weight_names(cfg: AlignAIRConfig) -> tuple[str, ...]:
    """Names only: validating a batch must not allocate training parameters."""
    keys = []
    if cfg.classification_mode == "multilabel":
        for g in genes_of(cfg):
            keys += [f"{g}_start", f"{g}_end", f"{g}_classification"]
    anchors = getattr(cfg, "anchor_heads", False)
    if cfg.use_meta_heads:
        keys += ["mutation", "indel"] + ([] if anchors else ["productive"])   # anchors: productivity is derived
    if cfg.use_orientation:
        keys.append("orientation")
    if cfg.use_meta_heads and getattr(cfg, "num_chain_types", 1) > 1:                  # multi-chain: chain_type (locus) head
        keys.append("chain_type")
    if getattr(cfg, "state_head", False):                      # per-position edit-state head
        keys.append("state")
    if getattr(cfg, "region_head", False):                     # per-position recombination-region head
        keys.append("regions")
    if anchors:                                                # the V and J anchor heads
        keys += [ANCHOR_KEYS[gene] for gene in ANCHOR_GENES]
    if getattr(cfg, "d_presence_head", False):                 # the 3-state D presence head
        keys.append(D_PRESENCE)
    return tuple(keys)



def make_loss_weights(cfg: AlignAIRConfig) -> nn.ModuleDict:
    """Create the trainable uncertainty weights once, alongside the model's optimizer."""
    return nn.ModuleDict({key: UncertaintyWeight() for key in loss_weight_names(cfg)})


def auxiliary_losses(out, targets, cfg, logvars):
    """Optional model heads share one objective implementation across score families."""
    parts = {}
    if cfg.use_meta_heads:
        mut = F.l1_loss(out["mutation_rate"], targets["mutation_rate"])
        ind = F.l1_loss(out["indel_count"], targets["indel_count"])
        parts["mutation"] = _kendall(logvars["mutation"], mut)
        parts["indel"] = _kendall(logvars["indel"], ind)
        if not getattr(cfg, "anchor_heads", False):                # with anchors, productivity is derived, not learned
            known = productive_rows(targets, out["productive"])    # unknown productivity is masked out
            prod = _mean_rows(F.binary_cross_entropy(out["productive"].clamp(1e-7, 1 - 1e-7), targets["productive"],
                                                     reduction="none"), known)
            parts["productive"] = _kendall(logvars["productive"], prod, known)

    # ---- per-position edit state: masked, class-weighted CE (germline/sub/insertion/deletion) ----
    if "state_logits" in out and "state_labels" in targets and "state" in logvars:
        sl = out["state_logits"]                                    # (B, L, S)
        w = sl.new_tensor(STATE_CLASS_WEIGHTS)
        labels = targets["state_labels"].reshape(-1)
        if (labels != _IGNORE).any():
            st_ce = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), labels,
                                    weight=w, ignore_index=_IGNORE)
            parts["state"] = _kendall(logvars["state"], st_ce)
        else:
            parts["state"] = sl.reshape(-1)[:0].sum() + logvars["state"].log_var * 0

    # ---- per-position recombination region (pre/V/N1/D/N2/J/post): CE over the read's positions only ----
    if getattr(cfg, "region_head", False):
        rl = out["region_logits"]                                   # (B, L, R)
        read = out["position_mask"] if "position_mask" in out else out["start_mask"]   # a read position is a start slot
        labels = targets["region_labels"].masked_fill(~read, _IGNORE).reshape(-1)
        if (labels != _IGNORE).any():
            region_ce = F.cross_entropy(rl.reshape(-1, rl.shape[-1]), labels, ignore_index=_IGNORE)
            parts["regions"] = _kendall(logvars["regions"], region_ce)
        else:
            parts["regions"] = rl.reshape(-1)[:0].sum() + logvars["regions"].log_var * 0

    # ---- V/J anchors: soft-label CE over the read's codon starts plus the null slot ("not observed") ----
    if getattr(cfg, "anchor_heads", False):
        read = out["position_mask"] if "position_mask" in out else out["start_mask"]
        starts = codon_starts(read)                                          # an anchor is a whole codon's start
        slots = torch.cat([starts, starts.new_ones(starts.shape[0], 1)], dim=-1)     # + the null slot
        for gene in ANCHOR_GENES:
            key = ANCHOR_KEYS[gene]
            rows = target_rows(targets, key, starts)                         # an unscorable anchor is left out
            ce = _soft_ce_per_sample(out[ANCHOR_LOGITS[gene]], anchor_target(targets[key], starts), slots)
            parts[key] = _kendall(logvars[key], _mean_rows(ce, rows), rows)

    # ---- D presence (absent / too short to call / callable): cross entropy over the three classes ----
    if getattr(cfg, "d_presence_head", False):
        logits = out[D_PRESENCE_LOGITS]
        rows = target_rows(targets, D_PRESENCE, logits)                      # an unscorable D is left out
        ce = F.cross_entropy(logits, targets[D_PRESENCE], reduction="none")
        parts[D_PRESENCE] = _kendall(logvars[D_PRESENCE], _mean_rows(ce, rows), rows)

    # ---- orientation (supervised only when the batch carries an orientation label) ----
    if "orientation" in targets and "orientation_logits" in out:
        o_loss = F.cross_entropy(out["orientation_logits"], targets["orientation"])
        parts["orientation"] = _kendall(logvars["orientation"], o_loss)

    # ---- chain_type / locus (multi-chain only; supervised when the batch carries the label) ----
    if "chain_type" in targets and "chain_type_logits" in out and "chain_type" in logvars:
        ct_loss = F.cross_entropy(out["chain_type_logits"], targets["chain_type"])
        parts["chain_type"] = _kendall(logvars["chain_type"], ct_loss)
    return parts



def hierarchical_loss(out: dict, targets: dict, cfg: AlignAIRConfig, logvars: nn.ModuleDict):
    L = cfg.max_seq_length
    genes = genes_of(cfg)
    parts: dict = {}
    row_masks = {gene: (supervised_rows(targets, gene, out[f"{gene}_allele"])
                        if any(f"{gene}_{suffix}" in targets for suffix in ("supervise", "valid")) else None)
                 for gene in genes}

    # ---- segmentation: soft-Gaussian CE per boundary (Kendall-weighted) + aux regularizers ----
    pos_mask = out.get("position_mask")                         # restrict CE to the read (not the pad)
    seg = out["v_start"].new_zeros(())
    for g in genes:
        supervise = row_masks[g]
        for b in ("start", "end"):
            key = f"{g}_{b}"
            mask = out.get(f"{b}_mask", pos_mask)
            target = (masked_soft_gaussian(targets[key], L, mask) if f"{b}_mask" in out
                      else soft_gaussian_target(targets[key], L))
            ce = _soft_ce(out[f"{key}_logits"], target, mask, supervise)
            seg = seg + _kendall(logvars[key], ce, supervise)
    aux_len = out["v_start"].new_zeros(())
    aux_iou = out["v_start"].new_zeros(())
    aux_hinge = out["v_start"].new_zeros(())
    for g in genes:
        supervise = row_masks[g]
        sp, ep = out[f"{g}_start"], out[f"{g}_end"]
        st, et = targets[f"{g}_start"].reshape(-1, 1), targets[f"{g}_end"].reshape(-1, 1)
        len_p, len_t = ep - sp, et - st
        aux_len = aux_len + _mean_rows(F.huber_loss(len_p, len_t, delta=1.0, reduction="none"), supervise)
        inter = (torch.minimum(ep, et) - torch.maximum(sp, st)).clamp(min=0)
        union = (len_p.clamp(min=0) + len_t.clamp(min=0) - inter).clamp(min=1e-6)
        aux_iou = aux_iou + _mean_rows(1.0 - (inter / union), supervise)
        aux_hinge = aux_hinge + _mean_rows(F.relu(1.0 - len_p), supervise)
    seg = seg + 0.1 * aux_len + 0.1 * aux_iou + 0.05 * aux_hinge
    parts["segmentation"] = seg

    # ---- classification: multi-label sigmoid BCE (label smoothing 0.1), Kendall-weighted ----
    clf = out["v_start"].new_zeros(())
    for g in genes:
        supervise = row_masks[g]
        y = targets[f"{g}_allele"]
        y_smooth = y * (1.0 - 0.1) + 0.05                       # Keras label_smoothing=0.1
        bce = _mean_rows(F.binary_cross_entropy(
            out[f"{g}_allele"].clamp(1e-7, 1 - 1e-7), y_smooth, reduction="none"), supervise)
        clf = clf + _kendall(logvars[f"{g}_classification"], bce, supervise)
    for spec in cfg.gene_specs:                                 # short-D degenerate-span penalty
        if spec.short_d_penalty:
            d_len = (out[f"{spec.name}_end"] - out[f"{spec.name}_start"]).reshape(-1)
            short_d_prob = out[f"{spec.name}_allele"][:, -1]
            clf = clf + _mean_rows((d_len < 5).float() * short_d_prob,
                                   row_masks[spec.name])
    parts["classification"] = clf

    total = parts["segmentation"] + parts["classification"]
    for key, value in auxiliary_losses(out, targets, cfg, logvars).items():
        parts[key] = value
        total = total + value
    return total, parts



def masked_soft_gaussian(gt: torch.Tensor, length: int, slot_mask: torch.Tensor,
                         sigma: float = 1.5) -> torch.Tensor:
    """Soft-Gaussian target restricted to the allowed boundary slots and RENORMALIZED.

    ``_soft_ce`` drops masked terms without renormalizing, so an un-renormalized target at an edge
    silently loses mass and is under-penalized (at the terminal slot, ~35-40% of it)."""
    t = soft_gaussian_target(gt, length, sigma=sigma) * slot_mask
    return t / t.sum(-1, keepdim=True).clamp_min(1e-8)


def anchor_target(anchor: torch.Tensor, slots: torch.Tensor, sigma: float = ANCHOR_SIGMA) -> torch.Tensor:
    """The soft target (B, L + 1) of anchors (B,) over the slots an anchor can take ``slots`` (B, L: the read's codon
    starts) plus the null slot L: an observed anchor (a position below L) is a Gaussian around it over those slots,
    renormalised, with no weight on the null slot; an anchor at the null slot is a one-hot on it."""
    length = slots.shape[-1]
    observed = (anchor < length).reshape(-1, 1)
    gaussian = masked_soft_gaussian(anchor.float().clamp(max=length - 1), length, slots, sigma=sigma)
    return torch.cat([gaussian * observed, (~observed).to(gaussian.dtype)], dim=-1)



def categorical_loss(out: dict, targets: dict, genes=("v", "j")):
    """Mean boundary CE plus mean categorical/compatible-set CE over supervised rows.

    Rows without supervision are selected out before logarithms and cross entropy. Empty compatible
    sets on excluded rows therefore cannot create infinity-times-zero losses or non-finite gradients.
    """
    length = out[f"{genes[0]}_start_logits"].shape[-1]
    seg_terms, clf_terms = [], []
    for gene in genes:
        logits = out[f"{gene}_class_logits"]
        valid = supervised_rows(targets, gene, logits)
        for boundary in ("start", "end"):
            slot_mask = out.get(f"{boundary}_mask", out.get("position_mask"))
            boundary_logits = out[f"{gene}_{boundary}_logits"][valid]
            if valid.any():
                mask = slot_mask[valid]
                target = masked_soft_gaussian(targets[f"{gene}_{boundary}"][valid], length, mask)
                seg_terms.append(_soft_ce_per_sample(boundary_logits, target, mask).mean())
            else:
                seg_terms.append(boundary_logits.sum())
        classified = categorical_rows(targets, gene, logits)
        selected = logits[classified]
        if not classified.any():
            clf_terms.append(selected.sum())
        elif f"{gene}_compatible" in targets:
            compatible = targets[f"{gene}_compatible"][classified]
            logp = torch.log_softmax(selected, dim=-1)
            clf_terms.append(-torch.logsumexp(logp.masked_fill(compatible == 0, float("-inf")), dim=-1).mean())
        else:
            clf_terms.append(F.cross_entropy(selected, targets[f"{gene}_class"][classified]))
    segmentation = torch.stack(seg_terms).mean()
    classification = torch.stack(clf_terms).mean()
    return segmentation + classification, {"segmentation": segmentation, "classification": classification}
