"""Pure functions used by the hierarchical loss."""
import torch
import torch.nn.functional as F


def soft_targets(gt: torch.Tensor, L: int, sigma: float = 1.5) -> torch.Tensor:
    """Gaussian soft-label distribution centred at the (rounded, clamped) gt index."""
    gt = torch.round(gt.to(torch.float32)).clamp(0.0, float(L - 1))
    positions = torch.arange(L, dtype=torch.float32, device=gt.device).unsqueeze(0)
    dist2 = (positions - gt) ** 2
    logits = -0.5 * dist2 / (sigma * sigma)
    return torch.softmax(logits, dim=-1)


def expectation_from_logits(logits: torch.Tensor, max_seq_length: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    pos = torch.arange(max_seq_length, dtype=torch.float32, device=logits.device).unsqueeze(0)
    return (probs * pos).sum(dim=-1, keepdim=True)


def soft_label_cross_entropy(target_probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mean soft-label cross-entropy: -sum(target * log_softmax(logits))."""
    log_probs = F.log_softmax(logits, dim=-1)
    return (-(target_probs * log_probs).sum(dim=-1)).mean()


def interval_iou_loss(s_pred: torch.Tensor, e_pred: torch.Tensor,
                      s_true: torch.Tensor, e_true: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    s_pred = s_pred.squeeze(-1)
    e_pred = e_pred.squeeze(-1)
    inter = torch.relu(torch.minimum(e_pred, e_true) - torch.maximum(s_pred, s_true))
    len_pred = torch.clamp(e_pred - s_pred, min=0.0)
    len_true = torch.clamp(e_true - s_true, min=0.0)
    union = len_pred + len_true - inter + eps
    iou = inter / union
    return 1.0 - iou.mean()
