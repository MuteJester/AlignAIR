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


def _soft_ce_per_sample(logits, soft_target, pos_mask):
    """Per-sample soft-label CE over positions (mean over the batch reproduces core `_soft_ce`)."""
    logits = logits.masked_fill(~pos_mask, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    term = soft_target * logp
    term = torch.where(pos_mask, term, term.new_zeros(()))
    return -term.sum(-1)                                       # (B,)


def aa_loss(out: dict, targets: dict, genes=("v", "j")):
    """Soft-Gaussian boundary CE + softmax class CE, averaged. A gene may carry a per-sample validity
    mask ``{g}_valid`` (D: 0 for inverted/absent) — its boundary terms are averaged over valid samples
    and its class CE ignores the -1 sentinel. V/J have no mask (all valid)."""
    L = out["v_start_logits"].shape[-1]
    masks = {"start": out["start_mask"], "end": out["end_mask"]}

    def _reduce(ps, valid):
        return ps.mean() if valid is None else (ps * valid).sum() / valid.sum().clamp_min(1.0)

    seg_terms, clf_terms = [], []
    for g in genes:
        valid = targets.get(f"{g}_valid")                     # (B,) float, or None (all valid)
        for b in ("start", "end"):
            m = masks[b]
            ps = _soft_ce_per_sample(out[f"{g}_{b}_logits"], masked_soft_gaussian(targets[f"{g}_{b}"], L, m), m)
            seg_terms.append(_reduce(ps, valid))
        logits = out[f"{g}_class_logits"]
        if f"{g}_compatible" in targets:                      # PARTIAL-LABEL: -log Σ_{c∈compatible} p[c]
            logp = torch.log_softmax(logits, dim=-1)
            comp = targets[f"{g}_compatible"]                 # (B, C) multi-hot; a singleton -> plain CE
            ps = -torch.logsumexp(logp.masked_fill(comp == 0, float("-inf")), dim=-1)
            clf_terms.append(_reduce(ps, valid))
        else:
            ce = F.cross_entropy(logits, targets[f"{g}_class"], ignore_index=-1, reduction="none")
            clf_terms.append(_reduce(ce, valid))

    seg = torch.stack(seg_terms).mean()
    clf = torch.stack(clf_terms).mean()
    return seg + clf, {"segmentation": seg, "classification": clf}
