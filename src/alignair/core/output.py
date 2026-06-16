"""Typed model output container."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Dict

import torch


@dataclass
class AlignAIROutput:
    v_start_logits: torch.Tensor
    v_end_logits: torch.Tensor
    j_start_logits: torch.Tensor
    j_end_logits: torch.Tensor
    v_start: torch.Tensor
    v_end: torch.Tensor
    j_start: torch.Tensor
    j_end: torch.Tensor
    v_allele: torch.Tensor
    j_allele: torch.Tensor
    mutation_rate: torch.Tensor
    indel_count: torch.Tensor
    productive: torch.Tensor
    # Optional D-gene fields.
    d_start_logits: Optional[torch.Tensor] = None
    d_end_logits: Optional[torch.Tensor] = None
    d_start: Optional[torch.Tensor] = None
    d_end: Optional[torch.Tensor] = None
    d_allele: Optional[torch.Tensor] = None
    # Optional multi-chain field.
    chain_type: Optional[torch.Tensor] = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name)
                for f in fields(self) if getattr(self, f.name) is not None}
