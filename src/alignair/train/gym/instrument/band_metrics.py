"""Gate-1 band-head metrics: recall (top-1 + top-m union), fail-open rate, and the
effective DP cell budget (speed proxy). Recall and budget are gated TOGETHER so a
broad/always-open predictor cannot pass recall while destroying speed (spec §5)."""
import torch


def top1_recall(offset_logits, true_start, w):
    pred = offset_logits.argmax(dim=-1)
    return float(((pred - true_start).abs() <= w).float().mean())


def _topm_centers(offset_logits, w, m):
    """Greedy NMS: repeatedly take the max, suppress +-2w around it, m times. (B,m)."""
    x = offset_logits.clone()
    B, Lg = x.shape
    pos = torch.arange(Lg, device=x.device)
    centers = []
    for _ in range(m):
        c = x.argmax(dim=-1)                                   # (B,)
        centers.append(c)
        near = (pos.unsqueeze(0) - c.unsqueeze(1)).abs() <= 2 * w
        x = x.masked_fill(near, -1e9)
    return torch.stack(centers, dim=1)                         # (B,m)


def topm_union_recall(offset_logits, true_start, w, m):
    centers = _topm_centers(offset_logits, w, m)               # (B,m)
    hit = ((centers - true_start.unsqueeze(1)).abs() <= w).any(dim=1)
    return float(hit.float().mean())


def fail_open_rate(offset_logits, threshold):
    maxp = torch.softmax(offset_logits.float(), dim=-1).max(dim=-1).values
    return float((maxp < threshold).float().mean())


def conf_fail_open_rate(conf_logit, conf_thresh):
    """Fail-open rate from the LEARNED confidence: fraction with sigmoid(conf) < threshold."""
    return float((torch.sigmoid(conf_logit) < conf_thresh).float().mean())


def committed_recall(offset_logits, conf_logit, true_start, w, m, conf_thresh):
    """Top-m union recall AMONG committed (non-fail-open) reads. A fail-open read goes to full
    DP (always correct) so it is excluded here, not counted as a miss (deployment-faithful, spec §5)."""
    committed = torch.sigmoid(conf_logit) >= conf_thresh
    if committed.sum() == 0:
        return 1.0
    centers = _topm_centers(offset_logits[committed], w, m)
    hit = ((centers - true_start[committed].unsqueeze(1)).abs() <= w).any(dim=1)
    return float(hit.float().mean())


def cell_budget(offset_logits, w, threshold, seg_len):
    Lg = offset_logits.shape[-1]
    maxp = torch.softmax(offset_logits.float(), dim=-1).max(dim=-1).values
    open_ = maxp < threshold
    cols = torch.where(open_, torch.full_like(seg_len, Lg), torch.full_like(seg_len, 2 * w + 1))
    return float((cols.float() * seg_len.float()).mean())
