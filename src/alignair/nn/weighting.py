"""Kendall-style uncertainty weighting (port of RegularizedConstrainedLogVar).

A trainable scalar log-variance; ``forward()`` returns the precision exp(-log_var)
used to weight a task loss. ``regularization()`` returns a soft penalty that
discourages very small log-variance. ``apply_constraints()`` clamps log_var to
[min_log_var, max_log_var] and is called by the trainer after each optimizer step.
"""
import math

import torch
import torch.nn as nn


class UncertaintyWeight(nn.Module):
    def __init__(self, initial_value: float = 1.0, min_log_var: float = -3.0,
                 max_log_var: float = 1.0, regularizer_weight: float = 0.01):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.regularizer_weight = regularizer_weight
        self.log_var = nn.Parameter(torch.tensor(math.log(initial_value), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        log_var = torch.clamp(self.log_var, self.min_log_var, self.max_log_var)
        return torch.exp(-log_var)

    def regularization(self) -> torch.Tensor:
        return self.regularizer_weight * torch.relu(-self.log_var - 2.0)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        self.log_var.clamp_(self.min_log_var, self.max_log_var)
