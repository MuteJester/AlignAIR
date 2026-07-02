"""Hierarchical multitask loss for SingleChainAlignAIR (PyTorch port).

Per boundary: soft-label cross-entropy (Gaussian target around the GT position) — segmentation-first
localization — plus small IoU/length/hinge auxiliaries. Per gene: multi-label allele BCE with label
smoothing. Plus mutation (L1), indel (L1), productivity (BCE). Every term is Kendall-weighted by a
learnable precision (RegularizedConstrainedLogVar). Faithful to legacy `hierarchical_loss`.
"""
import torch
import torch.nn.functional as F

from ..sota.matching import contrastive_match_loss

_SEG = ["v_start", "v_end", "j_start", "j_end"]


def _allele_loss(pred, target, g):
    """Dynamic mode -> multi-positive InfoNCE over the reference; fixed mode -> BCE + label smoothing."""
    if f"{g}_allele_scores" in pred:
        return contrastive_match_loss(pred[f"{g}_allele_scores"], target[f"{g}_allele"])
    p = pred[f"{g}_allele"].clamp(1e-6, 1 - 1e-6)
    t = target[f"{g}_allele"] * 0.9 + 0.05                   # label smoothing 0.1
    return F.binary_cross_entropy(p, t)


def _soft_targets(gt, L, sigma=1.5):                        # gt (B,) -> (B, L) gaussian softmax
    gt = gt.round().clamp(0, L - 1).reshape(-1, 1)
    pos = torch.arange(L, device=gt.device, dtype=torch.float32)[None]
    return (-0.5 * (pos - gt) ** 2 / (sigma * sigma)).softmax(-1)


def _ce(soft_target, logits):
    return -(soft_target * logits.log_softmax(-1)).sum(-1).mean()


def _iou_loss(sp, ep, st, et, eps=1e-6):
    inter = F.relu(torch.minimum(ep, et) - torch.maximum(sp, st))
    union = F.relu(ep - sp) + F.relu(et - st) - inter + eps
    return 1.0 - (inter / union).mean()


def hierarchical_loss(model, pred: dict, target: dict) -> tuple:
    L, has_d = model.L, model.has_d
    logs, reg_total = {}, pred["v_start_logits"].new_zeros(())
    seg_keys = _SEG + (["d_start", "d_end"] if has_d else [])

    def weighted(key, loss):
        prec, reg = model.log_vars[key]()
        nonlocal reg_total
        reg_total = reg_total + reg
        logs[key] = float(loss.detach())
        return loss * prec

    # --- soft-label boundary CE ---
    seg_loss = pred["v_start_logits"].new_zeros(())
    exp = {}
    for s in seg_keys:
        exp[s] = model._expectation(pred[f"{s}_logits"]).squeeze(-1)
        seg_loss = seg_loss + weighted(s, _ce(_soft_targets(target[s].float(), L), pred[f"{s}_logits"]))

    # --- auxiliary segmentation (len Huber, IoU, hinge) ---
    genes = [("v", "v_start", "v_end"), ("j", "j_start", "j_end")]
    if has_d:
        genes.append(("d", "d_start", "d_end"))
    len_loss = iou_loss = hinge = pred["v_start_logits"].new_zeros(())
    for _, sk, ek in genes:
        lp = exp[ek] - exp[sk]
        lt = target[ek].float() - target[sk].float()
        len_loss = len_loss + F.huber_loss(lp, lt, delta=1.0)
        iou_loss = iou_loss + _iou_loss(exp[sk], exp[ek], target[sk].float(), target[ek].float())
        hinge = hinge + F.relu(1.0 - lp).mean()
    seg_loss = seg_loss + 0.1 * len_loss + 0.1 * iou_loss + 0.05 * hinge
    logs["seg_aux"] = float((0.1 * len_loss + 0.1 * iou_loss + 0.05 * hinge).detach())

    # --- allele identity (fixed: BCE; dynamic: InfoNCE over the reference) ---
    clf = weighted("v_clf", _allele_loss(pred, target, "v")) \
        + weighted("j_clf", _allele_loss(pred, target, "j"))
    if has_d:
        clf = clf + weighted("d_clf", _allele_loss(pred, target, "d"))

    # --- meta ---
    mut = weighted("mutation", F.l1_loss(pred["mutation_rate"].squeeze(-1), target["mutation_rate"].float()))
    ind = weighted("indel", F.l1_loss(pred["indel_count"].squeeze(-1), target["indel_count"].float()))
    prod = weighted("productivity",
                    F.binary_cross_entropy(pred["productive"].squeeze(-1).clamp(1e-6, 1 - 1e-6),
                                           target["productive"].float()))

    total = seg_loss + clf + mut + ind + prod + reg_total
    logs["total"] = float(total.detach())
    logs["segmentation"], logs["classification"] = float(seg_loss.detach()), float(clf.detach())
    return total, logs
