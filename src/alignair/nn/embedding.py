"""Token + learned position embedding (port of legacy TokenAndPositionEmbedding)."""
import torch
import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    """Sum of a token embedding and a learned position embedding.

    Input:  (B, L) integer tokens in [0, vocab_size).
    Output: (B, L, embed_dim).
    """

    def __init__(self, max_len: int, vocab_size: int = 6, embed_dim: int = 32):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.int32, torch.int64):
            x = x.long()
        positions = torch.arange(self.max_len, device=x.device)
        return self.token_emb(x) + self.pos_emb(positions)
