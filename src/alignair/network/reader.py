"""The SHM reader: what happened to each base, read from its sequence context alone.

Per base, from the trunk's features: the edit state (``READ_EDITS``: germline, substitution, insertion), whose
substitution probability ``m`` and insertion probability ``inserted`` the bank scans use as costs; and per gene the
base's column in that gene's frame (``alignair.reference.frames``). It never predicts the germline base itself, which
would be the reference memorized: what the germline is comes only from the bank. Both heads are 1x1 convolutions of
the trunk's features; the frames are fixed, so the weights do not depend on the reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..core.edit_states import EDIT_LABELS
from ..reference.frames import FRAME_WIDTHS

# The edit labels a read base can carry: every one but deletion (no read base), which is the last label, so the
# reader's class indices are the training labels' (``core.edit_states.EDIT_INDEX``).
assert EDIT_LABELS[-1] == "deletion", "deletion must be the last edit label"
READ_EDITS = EDIT_LABELS[:-1]


@dataclass
class ReaderOutput:
    """The reader's per-base edit logits and, per gene, column logits."""
    edit_logits: torch.Tensor                 # (B, L, 3) over READ_EDITS
    coord_logits: dict                        # gene -> (B, L, FRAME_WIDTHS[gene])

    @property
    def m(self) -> torch.Tensor:
        """(B, L) each base's substitution probability."""
        return self.edit_logits.softmax(-1)[..., 1]

    @property
    def inserted(self) -> torch.Tensor:
        """(B, L) each base's insertion probability."""
        return self.edit_logits.softmax(-1)[..., 2]


def top_coordinates(coord_logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each base's ``k`` most probable columns (B, L, k) and their softmax probabilities, not renormalized: the mass
    the reader puts elsewhere stays out of the scan."""
    probs, columns = coord_logits.softmax(-1).topk(k, dim=-1)
    return columns, probs


class ShmReader(nn.Module):
    """The SHM reader (see the module docstring): an edit head and a column head per gene."""

    def __init__(self, channels: int, genes: tuple[str, ...] = ("v", "d", "j")):
        super().__init__()
        self.edit = nn.Conv1d(channels, len(READ_EDITS), 1)
        self.coords = nn.ModuleDict({gene: nn.Conv1d(channels, FRAME_WIDTHS[gene], 1) for gene in genes})

    def forward(self, features: torch.Tensor, mask: torch.Tensor, coordinates: bool = True) -> ReaderOutput:
        """The edit logits and, with ``coordinates``, every base's column logits for every gene."""
        x = features.transpose(1, 2)
        return ReaderOutput(self.edit(x).transpose(1, 2),
                            {gene: head(x).transpose(1, 2) for gene, head in self.coords.items()} if coordinates
                            else {})

    def top_columns(self, features: torch.Tensor, gene: str, support: torch.Tensor,
                    k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``top_coordinates`` of ``gene`` (B, L, k) read only over each read's window from its first to its last
        ``support``ed base (B, L): the columns there are the whole read's; outside the window, and in a read with no
        supported base, the probabilities are 0. A gene's bases are a small part of a read (D and J most of all), so
        this skips most of the column logits and their top-k."""
        batch, length, channels = features.shape
        position = torch.arange(length, device=features.device)
        first = torch.where(support, position, length).amin(1)
        span = (torch.where(support, position, -1).amax(1) - first + 1).clamp_min(0)
        columns = torch.zeros(batch, length + 1, k, dtype=torch.long, device=features.device)
        probs = features.new_zeros(batch, length + 1, k)
        width = int(span.max())
        if width:
            offset = torch.arange(width, device=features.device)
            index = (first[:, None] + offset).clamp_max(length - 1)                        # (B, width)
            window = features.gather(1, index.unsqueeze(-1).expand(-1, -1, channels))
            head = self.coords[gene]
            top, weight = top_coordinates(F.linear(window, head.weight[..., 0], head.bias), k)
            slot = torch.where(offset < span[:, None], index, length).unsqueeze(-1).expand(-1, -1, k)  # past: dropped
            columns.scatter_(1, slot, top)
            probs.scatter_(1, slot, weight)
        return columns[:, :length], probs[:, :length]
