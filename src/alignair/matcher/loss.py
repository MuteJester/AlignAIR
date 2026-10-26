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


def selection_ce_batch(scores, target_mask, tau: float, margin: float, reduce: bool = True):
    """Vectorized additive-margin, set-aware CE over candidate subsets.

    scores and target_mask are [B,K]. Rows with no positive target are ignored.
    This is the loss shape used by sampled truth+sibling+in-batch training, where K is
    small and independent of the full reference size.
    """
    valid = target_mask.any(dim=-1)
    logits = (scores - margin * target_mask.float()) / tau
    denom = torch.logsumexp(logits, dim=-1)
    num = torch.logsumexp(logits.masked_fill(~target_mask, -1e9), dim=-1)
    loss = denom - num
    loss = loss[valid]
    if not reduce:
        return loss
    return loss.mean() if loss.numel() else scores.sum() * 0.0
