"""A learned loss weight for the training-engine tests: one trainable log-variance, clamped by the engine's
``apply_constraints`` call after each optimizer step (the hook a task weight may define)."""
import torch
from torch import nn


class ClampedLogVar(nn.Module):
    def __init__(self, min_log_var: float = -3.0, max_log_var: float = 3.0):
        super().__init__()
        self.min_log_var, self.max_log_var = min_log_var, max_log_var
        self.log_var = nn.Parameter(torch.tensor(0.))

    def forward(self) -> torch.Tensor:
        return torch.exp(-self.log_var)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        self.log_var.clamp_(self.min_log_var, self.max_log_var)
