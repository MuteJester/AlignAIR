"""Running-mean accumulator (mirrors keras.metrics.Mean)."""
import torch


class MeanAccumulator:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: torch.Tensor) -> None:
        v = value.detach()
        self.total += float(v.sum().item())
        self.count += int(v.numel())

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.total / self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
