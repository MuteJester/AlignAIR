"""The SHM reader: what happened to each base, read from its sequence context alone.

Per base, from the trunk's features: the edit state (``EDIT_STATES``: germline, substitution, insertion), whose
substitution probability ``m`` and insertion probability ``inserted`` the bank scans use as costs; and per gene the
base's column in that gene's frame (``alignair.reference.frames``). It never predicts the germline base itself, which
would be the reference memorized: what the germline is comes only from the bank. Both heads are 1x1 convolutions of
the trunk's features; the frames are fixed, so the weights do not depend on the reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from alignair.reference.frames import FRAME_WIDTHS

from ..trunk import TRUNK_CHANNELS

EDIT_STATES = ("germline", "substitution", "insertion")


@dataclass
class ReaderOutput:
    edit_logits: torch.Tensor                 # (B, L, 3) over EDIT_STATES
    coord_logits: dict                        # gene -> (B, L, FRAME_WIDTHS[gene])

    @property
    def m(self) -> torch.Tensor:
        return self.edit_logits.softmax(-1)[..., 1]

    @property
    def inserted(self) -> torch.Tensor:
        return self.edit_logits.softmax(-1)[..., 2]


def top_coordinates(coord_logits: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Each base's ``k`` most probable columns (B, L, k) and their softmax probabilities, not renormalized: the mass
    the reader puts elsewhere stays out of the scan."""
    probs, columns = coord_logits.softmax(-1).topk(k, dim=-1)
    return columns, probs


class ShmReader(nn.Module):
    def __init__(self, channels: int = TRUNK_CHANNELS, genes: tuple[str, ...] = ("v", "d", "j")):
        super().__init__()
        self.edit = nn.Conv1d(channels, len(EDIT_STATES), 1)
        self.coords = nn.ModuleDict({gene: nn.Conv1d(channels, FRAME_WIDTHS[gene], 1) for gene in genes})

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> ReaderOutput:
        x = features.transpose(1, 2)
        return ReaderOutput(self.edit(x).transpose(1, 2),
                            {gene: head(x).transpose(1, 2) for gene, head in self.coords.items()})
