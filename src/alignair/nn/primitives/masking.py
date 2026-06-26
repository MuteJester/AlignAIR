"""Differentiable soft cutout mask (port of legacy SoftCutoutLayer).

Produces a length-``max_size`` mask per sample from start/end position
expectations using smooth sigmoid ramps, so gradients flow through the predicted
boundaries. Semantics: start-inclusive, end-exclusive [start, end), with
end >= start + 1 and bounds clamped to [0, max_size]."""
import torch
import torch.nn as nn


class SoftCutout(nn.Module):
    def __init__(self, max_size: int, k: float = 3.0):
        super().__init__()
        self.max_size = int(max_size)
        self.k = float(k)

    def _sanitize(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32).reshape(-1, 1)
        return torch.clamp(t, 0.0, float(self.max_size))

    def forward(self, start_raw: torch.Tensor, end_raw: torch.Tensor) -> torch.Tensor:
        start = self._sanitize(start_raw)
        end = self._sanitize(end_raw)
        end = torch.maximum(end, start + 1.0)

        indices = torch.arange(self.max_size, dtype=torch.float32,
                               device=start.device).unsqueeze(0)  # (1, L)
        left = torch.sigmoid((indices - start) / self.k)
        right = torch.sigmoid((end - indices) / self.k)
        return left * right  # (B, L)
