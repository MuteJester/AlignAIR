"""Stage-4 Task 4.1 — six-term guard loss with objective-specific masks, an energy margin, a supervised
contrastive term, and a source-adversarial term via gradient reversal.

    total = L_type + λ_complete·L_complete + λ_app·L_applicability
            + λ_oe·L_energy_margin + λ_contrast·L_supcon + λ_source·L_source_adv

Safeguard 2 (objective-specific masks): every sample contributes ONLY to losses for which it has a genuine
label. OE samples never receive an invented domain/completeness label; structural/known-negative/OE/positive
each carry masks per objective.

Safeguard 3 (gradient reversal, correct): grad_reverse sits between the shared encoder representation and the
source classifier. In backward it reverses the gradient flowing INTO the representation (so the encoder is
pushed to REMOVE source information) while the source-classifier parameters receive NORMAL gradients (they
learn to predict source). We reverse the gradient on the REPRESENTATION, never on the classifier params.

Safeguard 4 (source/category confounding): the source-adversarial term is applied only within target strata
holding >=2 meaningful sources. Where locus and source are perfectly confounded, the caller sets that
stratum's source mask to 0 so C4 cannot erase the locus signal.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)                    # identity forward

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None    # reverse the grad entering the representation


def grad_reverse(x, lambd: float = 1.0):
    """Apply between the shared encoder rep and the source classifier. Forward: identity. Backward: the grad
    into `x` (the representation) is reversed; the classifier's own params get normal gradients."""
    return _GradReverse.apply(x, lambd)


def _masked_ce(logits, target, mask):
    if mask is None or mask.sum() == 0:
        return logits.sum() * 0.0
    idx = mask.bool()
    return F.cross_entropy(logits[idx], target[idx].long())


def energy_margin_loss(energy, is_outlier, mask, m_in: float = 5.0, m_out: float = 10.0):
    """In-domain energy pushed below m_in; auxiliary outliers (OE) pushed above m_out. `energy` is the head's
    free energy = -logsumexp(domain_type_logits). Masked to samples with a genuine energy label."""
    if mask is None or mask.sum() == 0:
        return energy.sum() * 0.0
    m = mask.bool()
    e, out = energy[m], is_outlier[m].bool()
    loss = torch.zeros((), device=energy.device)
    if (~out).any():
        loss = loss + F.relu(e[~out] - m_in).mean()          # in-domain: energy < m_in
    if out.any():
        loss = loss + F.relu(m_out - e[out]).mean()          # outliers:  energy > m_out
    return loss


def supcon_loss(rep, labels, mask, temperature: float = 0.1):
    """Supervised contrastive loss over the L2-normalized representation, restricted to labeled samples.
    Computed in fp32 with a FINITE self-mask (-1e4, not -inf) so it is numerically stable under AMP fp16
    (an -inf self-mask makes logsumexp produce inf-inf -> NaN)."""
    if mask is None or mask.sum() < 2:
        return rep.sum() * 0.0
    idx = mask.bool()
    z = F.normalize(rep[idx].float(), dim=-1)
    y = labels[idx].long()
    sim = z @ z.t() / temperature
    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e4)                     # finite -> stable logsumexp under AMP
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    denom = pos.sum(1).clamp(min=1)
    loss = -((logp * pos).sum(1) / denom)
    finite = torch.isfinite(loss)
    return loss[finite].mean() if finite.any() else rep.sum() * 0.0


def guard_loss(outputs, targets, weights, masks):
    """outputs: {domain_type, completeness, applicability, source_logits, energy, rep}
    targets: {domain_type, completeness, applicability, source, energy_is_outlier, contrast}
    masks:   {type, complete, app, energy, contrast, source}  (per-objective; Safeguard 2 + 4)
    weights: {complete, app, oe, contrast, source}  -> returns (total, parts)."""
    parts = {}
    parts["type"] = _masked_ce(outputs["domain_type"], targets["domain_type"], masks.get("type"))
    parts["complete"] = _masked_ce(outputs["completeness"], targets["completeness"], masks.get("complete"))

    app_mask = masks.get("app")
    if app_mask is not None and app_mask.sum() > 0:
        i = app_mask.bool()
        parts["app"] = F.binary_cross_entropy_with_logits(
            outputs["applicability"][i].squeeze(-1), targets["applicability"][i].float())
    else:
        parts["app"] = outputs["applicability"].sum() * 0.0

    parts["energy"] = energy_margin_loss(outputs["energy"], targets["energy_is_outlier"], masks.get("energy"))
    parts["contrast"] = supcon_loss(outputs["rep"], targets["contrast"], masks.get("contrast"))
    parts["source"] = _masked_ce(outputs["source_logits"], targets["source"], masks.get("source"))

    total = (parts["type"]
             + weights.get("complete", 1.0) * parts["complete"]
             + weights.get("app", 1.0) * parts["app"]
             + weights.get("oe", 1.0) * parts["energy"]
             + weights.get("contrast", 1.0) * parts["contrast"]
             + weights.get("source", 1.0) * parts["source"])
    return total, parts
