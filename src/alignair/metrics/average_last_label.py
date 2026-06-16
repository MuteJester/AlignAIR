"""Mean of the last allele column (the 'Short-D' probability)."""
import torch

from .accumulator import MeanAccumulator


class AverageLastLabel(MeanAccumulator):
    def update(self, d_allele: torch.Tensor) -> None:
        super().update(d_allele[:, -1])
