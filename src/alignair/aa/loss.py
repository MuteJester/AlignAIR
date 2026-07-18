"""AA loss: soft-Gaussian boundary CE (per-boundary slot masks, renormalized targets) + softmax
single-label class CE. Components are averaged (4 boundary terms, 2 class terms) so segmentation
does not swamp classification at init. D is absent in Phase 1."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.losses import _soft_ce, soft_gaussian_target


def masked_soft_gaussian(gt: torch.Tensor, length: int, slot_mask: torch.Tensor,
                         sigma: float = 1.5) -> torch.Tensor:
    """Soft-Gaussian target restricted to the allowed boundary slots and RENORMALIZED.

    ``_soft_ce`` drops masked terms without renormalizing, so an un-renormalized target at an edge
    silently loses mass and is under-penalized (at the terminal slot, ~35-40% of it)."""
    t = soft_gaussian_target(gt, length) * slot_mask
    return t / t.sum(-1, keepdim=True).clamp_min(1e-8)


def aa_loss(out: dict, targets: dict, genes=("v", "j")):
    L = out["v_start_logits"].shape[-1]
    masks = {"start": out["start_mask"], "end": out["end_mask"]}

    seg_terms = []
    for g in genes:
        for b in ("start", "end"):
            m = masks[b]
            tgt = masked_soft_gaussian(targets[f"{g}_{b}"], L, m)
            seg_terms.append(_soft_ce(out[f"{g}_{b}_logits"], tgt, m))
    clf_terms = [F.cross_entropy(out[f"{g}_class_logits"], targets[f"{g}_class"]) for g in genes]

    seg = torch.stack(seg_terms).mean()
    clf = torch.stack(clf_terms).mean()
    return seg + clf, {"segmentation": seg, "classification": clf}
