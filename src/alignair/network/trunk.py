"""The trunk: every read position's features, computed once per forward pass, and the anchor heads that read them.

``DilatedTrunk`` maps the orientation-corrected embedding (B, L, E) to per-position features (B, L, C): a 1x1 input
projection, then pre-norm residual blocks of same-padded dilated convolutions (``DILATIONS``, kernel 5), pads zeroed
so a read's features depend neither on its padding nor on how much there is. ``AnchorHead`` reads one conserved
anchor's start (the V Cys, the J Trp/Phe) or that the read does not show it.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# Two passes of doubling dilations; with kernel 5 each position sees 1 + 4 x 126 = 505 positions.
DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)


class _ResidualBlock(nn.Module):
    """``x + Dropout(GELU(Conv1d_dilated(LayerNorm(x) * keep)))`` on (B, L, C), keeping the length. LayerNorm
    normalizes each position over its channels, so no statistic is shared between positions or reads; ``keep``
    (B, L, 1) zeroes the normalized pad positions, where LayerNorm would leave its bias, before the convolution reads
    them."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding="same")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        h = self.conv((self.norm(x) * keep).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(h))


class DilatedTrunk(nn.Module):
    """``(emb (B, L, E), mask (B, L)) -> (B, L, channels)``: a 1x1 input projection, then pre-norm residual blocks of
    same-padded dilated convolutions. Pad positions (``mask`` False) are zeroed after the projection, after each
    block's normalization and after every block, so a convolution reads a pad position as it reads the zero padding
    past the window: a read's features depend neither on what its padding holds nor on how much padding there is.
    Each position sees ``receptive_field = 1 + (kernel - 1) * sum(dilations)`` positions around it, clipped to the
    read."""

    def __init__(self, in_channels: int, channels: int, kernel: int = 5,
                 dilations: tuple[int, ...] = DILATIONS, dropout: float = 0.1):
        super().__init__()
        self.project = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList(_ResidualBlock(channels, kernel, dilation, dropout) for dilation in dilations)
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keep = mask.unsqueeze(-1).to(emb.dtype)
        x = self.project(emb.transpose(1, 2)).transpose(1, 2) * keep
        for block in self.blocks:
            x = block(x, keep) * keep
        return x


class AnchorHead(nn.Module):
    """One conserved anchor's position, or that the read does not show it: ``(features (B, L, C), mask (B, L)) ->
    logits (B, L + 1)``. Slot k < L, the anchor codon starting at base k, is a 1x1 convolution of position k's
    features; slot L, the null slot ("not observed"), is a linear map of the read's mean and max features over its
    positions (``mask``). A pad position's slot is the convolution's bias (its features are zero); the anchor term
    and the decoder read the read's positions and the null slot only."""

    def __init__(self, channels: int):
        super().__init__()
        self.position = nn.Conv1d(channels, 1, 1)
        self.null = nn.Linear(2 * channels, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = self.position(features.transpose(1, 2))[:, 0]
        return torch.cat([positions, self.null(_pooled(features, mask))], -1)


def _pooled(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The read's mean and max features over its positions (``mask``): (B, L, C) -> (B, 2C); 0 for a read with no
    position."""
    keep = mask.unsqueeze(-1)
    mean = (features * keep).sum(1) / keep.sum(1).clamp_min(1)
    peak = features.masked_fill(~keep, float("-inf")).amax(1)
    peak = torch.where(mask.any(1, keepdim=True), peak, torch.zeros_like(peak))     # a read with no position
    return torch.cat([mean, peak], -1)
