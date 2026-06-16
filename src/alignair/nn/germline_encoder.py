"""GermlineEncoder: nucleotide sequence -> L2-normalized embedding (for matching)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GermlineEncoder(nn.Module):
    """Conv stack + masked mean-pool + projection -> unit-norm embedding.

    Input:  tokens (B, L) long, mask (B, L) bool (True = valid position).
    Output: (B, embed_dim) L2-normalized.
    """

    def __init__(self, embed_dim: int = 128, vocab_size: int = 6,
                 n_conv: int = 3, kernel: int = 5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, kernel, padding="same") for _ in range(n_conv)])
        self.act = nn.GELU()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)                 # (B, L, E)
        m = mask.unsqueeze(-1).to(x.dtype)         # (B, L, 1)
        x = x * m                                  # zero padded positions pre-conv
        h = x.transpose(1, 2)                      # (B, E, L)
        for conv in self.convs:
            h = self.act(conv(h)) * m.transpose(1, 2)  # re-mask after each conv
        h = h.transpose(1, 2)                      # (B, L, E)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # masked mean
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)
