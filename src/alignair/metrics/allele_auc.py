"""Multi-label AUC via the rank-statistic (Mann-Whitney U) formulation."""
import torch

from .accumulator import MeanAccumulator


class MultiLabelAUC:
    def __init__(self):
        self._mean = MeanAccumulator()

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        labels = (y_true.reshape(-1) > 0.5)
        scores = y_pred.reshape(-1)
        pos = scores[labels]
        neg = scores[~labels]
        if pos.numel() == 0 or neg.numel() == 0:
            return
        # AUC = P(score_pos > score_neg); count ties as 0.5.
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        wins = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
        auc = wins / (pos.numel() * neg.numel())
        self._mean.update(auc.reshape(1))

    def compute(self) -> torch.Tensor:
        return self._mean.compute()

    def reset(self) -> None:
        self._mean.reset()
