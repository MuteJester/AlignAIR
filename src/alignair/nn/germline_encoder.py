"""GermlineEncoder: nucleotide sequence -> per-position reps and a pooled
L2-normalized embedding (for matching)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GermlineEncoder(nn.Module):
    """Conv stack producing per-position reps; masked mean-pool + proj -> unit-norm embedding.

    Input:  tokens (B, L) long, mask (B, L) bool (True = valid position).
    forward_positions -> (B, L, embed_dim) masked per-position reps.
    forward -> (B, embed_dim) L2-normalized pooled embedding.
    """

    def __init__(self, embed_dim: int = 128, vocab_size: int = 6,
                 n_conv: int = 3, kernel: int = 5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, kernel, padding="same") for _ in range(n_conv)])
        self.act = nn.GELU()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward_positions(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)                 # (B, L, E)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        h = x.transpose(1, 2)
        for conv in self.convs:
            h = self.act(conv(h)) * m.transpose(1, 2)
        return h.transpose(1, 2)                    # (B, L, E), padded positions zero

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.forward_positions(tokens, mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)
