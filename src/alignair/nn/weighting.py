"""Kendall-style uncertainty weighting.

A trainable scalar log-variance ``s``. ``forward()`` returns the precision
``exp(-s)`` that weights a task loss. ``penalty()`` returns the Kendall balancing
term ``0.5 * s`` that MUST be added to the total loss — without it ``exp(-s)`` has
no upward pressure, so every positive task loss drives ``s`` to its clamp and all
weights collapse to the floor (the previous port omitted this term, leaving the
"learned balancing" inert). With the penalty the equilibrium precision is ~0.5/L,
i.e. tasks are automatically balanced inversely to their loss magnitude.
``apply_constraints()`` clamps ``s`` after each optimizer step.
"""
import math

import torch
import torch.nn as nn


class UncertaintyWeight(nn.Module):
    def __init__(self, initial_value: float = 1.0, min_log_var: float = -3.0,
                 max_log_var: float = 3.0, regularizer_weight: float = 0.01):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.regularizer_weight = regularizer_weight
        self.log_var = nn.Parameter(torch.tensor(math.log(initial_value), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(-torch.clamp(self.log_var, self.min_log_var, self.max_log_var))

    def penalty(self) -> torch.Tensor:
        """Kendall regularizer 0.5*s; balances the exp(-s) precision in the total."""
        return 0.5 * torch.clamp(self.log_var, self.min_log_var, self.max_log_var)

    def regularization(self) -> torch.Tensor:
        """Deprecated (legacy hierarchical loss only): soft floor penalty. The new
        DNAlignAIR loss uses penalty() for correct Kendall balancing instead."""
        return self.regularizer_weight * torch.relu(-self.log_var - 2.0)

    @torch.no_grad()
    def weight(self) -> float:
        return float(self.forward())

    @torch.no_grad()
    def apply_constraints(self) -> None:
        self.log_var.clamp_(self.min_log_var, self.max_log_var)
