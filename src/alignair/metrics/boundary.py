"""Boundary accuracy/MAE metrics from per-position logits + ground-truth scalar."""
import torch

from .accumulator import MeanAccumulator


class BoundaryMetrics:
    def __init__(self):
        self.mae = MeanAccumulator()
        self.acc = MeanAccumulator()
        self.acc_1nt = MeanAccumulator()

    def update(self, gt_scalar: torch.Tensor, logits: torch.Tensor) -> None:
        max_idx = logits.shape[-1] - 1
        gt_idx = torch.round(gt_scalar.squeeze(-1)).long().clamp(0, max_idx)
        pred_idx = logits.argmax(dim=-1)
        err = (pred_idx - gt_idx).abs().float()
        self.mae.update(err)
        self.acc.update((pred_idx == gt_idx).float())
        self.acc_1nt.update((err <= 1.0).float())

    def compute(self) -> dict:
        return {"mae": self.mae.compute().item(),
                "acc": self.acc.compute().item(),
                "acc_1nt": self.acc_1nt.compute().item()}

    def reset(self) -> None:
        self.mae.reset(); self.acc.reset(); self.acc_1nt.reset()
