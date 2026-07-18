"""Prediction-entropy metric for an allele probability head."""
import torch

from .accumulator import MeanAccumulator


class AlleleEntropy(MeanAccumulator):
    def update(self, probs: torch.Tensor) -> None:
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        super().update(entropy)
