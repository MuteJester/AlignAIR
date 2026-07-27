"""Stage-3 Task 3.3 — GuardHeads: completeness (5-way), domain_type (8-way), applicability (binary logit),
and an open-set free-energy over the recognized in-domain type logits (-logsumexp). Input is the fused
representation concat(cls, mean, [normalized HMM features]).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from alignair.aa.guard.classes import COMPLETENESS_CLASSES, DOMAIN_TYPE_CLASSES

N_COMPLETENESS = len(COMPLETENESS_CLASSES)      # frozen ordered semantics (classes.py)
N_DOMAIN_TYPE = len(DOMAIN_TYPE_CLASSES)        # recognized in-domain types for the open-set energy


class GuardHeads(nn.Module):
    def __init__(self, in_dim: int, n_completeness: int = N_COMPLETENESS, n_domain_type: int = N_DOMAIN_TYPE):
        super().__init__()
        self.completeness = nn.Linear(in_dim, n_completeness)
        self.domain_type = nn.Linear(in_dim, n_domain_type)     # recognized in-domain types
        self.applicability = nn.Linear(in_dim, 1)               # binary applicability logit

    def forward(self, x):
        return {"completeness": self.completeness(x),
                "domain_type": self.domain_type(x),
                "applicability": self.applicability(x)}

    @staticmethod
    def energy(domain_type_logits: torch.Tensor) -> torch.Tensor:
        """Open-set free energy over the recognized-type logits: E = -logsumexp(type_logits). Low energy =
        confidently a recognized in-domain type; energy DECREASES as any known-type logit grows."""
        return -torch.logsumexp(domain_type_logits, dim=-1)
