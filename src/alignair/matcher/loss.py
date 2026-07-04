import torch


def selection_ce(scores, target_mask, tau: float, margin: float):
    """Additive-margin, set-aware selection cross-entropy over one gene's unique sequences.

    scores [C], target_mask [C] bool (>=1 True). Margin is subtracted from the true
    logit(s) in SCORE space, before the tau scale: logits = (scores - m*mask)/tau.
    Numerator = logsumexp over the true set (any-correct).
    """
    logits = (scores - margin * target_mask.float()) / tau
    denom = torch.logsumexp(logits, dim=-1)
    num = torch.logsumexp(logits.masked_fill(~target_mask, -1e9), dim=-1)
    return denom - num
