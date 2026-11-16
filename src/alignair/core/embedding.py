"""The read's token embedding: a learned token embedding plus a learned absolute position embedding, added."""
from __future__ import annotations

import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    """Learned token embedding + learned absolute position embedding, added."""

    def __init__(self, vocab_size: int, embed_dim: int, maxlen: int):
        super().__init__()
        self.maxlen = maxlen
        self.token = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(maxlen, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L) int
        positions = torch.arange(self.maxlen, device=x.device)
        return self.token(x.long()) + self.pos(positions)
