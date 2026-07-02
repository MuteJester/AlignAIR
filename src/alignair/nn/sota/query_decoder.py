"""Typed VDJ object-query decoder.

A DETR-style transformer decoder (Facebook DETR, Apache-2.0): a small set of learned object
queries cross-attend the encoder memory (here, the fused read tokens) to produce one
representation per object. We use FIXED TYPED queries — one each for V, D, J — so roles are
fixed by the recombination structure and no Hungarian matching is needed. Built on
`torch.nn.MultiheadAttention` (the same attention DETR uses) with DETR's post-norm decoder
layer (self-attn among queries → cross-attn to memory → FFN). See sota/ATTRIBUTION.md.
"""
import torch
import torch.nn as nn

GENES = ("V", "D", "J")


class _DecoderLayer(nn.Module):
    def __init__(self, d: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, dim_ff), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(dim_ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, mem, mem_key_padding_mask=None):
        q = self.n1(q + self.drop(self.self_attn(q, q, q)[0]))                       # queries coordinate
        q = self.n2(q + self.drop(self.cross_attn(q, mem, mem,
                                                  key_padding_mask=mem_key_padding_mask)[0]))  # read
        q = self.n3(q + self.drop(self.ffn(q)))
        return q


class TypedVDJDecoder(nn.Module):
    """Three fixed typed queries (V, D, J) decode the fused read into one representation per gene."""

    def __init__(self, d_model: int, nhead: int = 8, n_layers: int = 3, dim_ff: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        dim_ff = dim_ff or 4 * d_model
        self.queries = nn.Parameter(torch.randn(len(GENES), d_model) * 0.02)   # V, D, J
        self.layers = nn.ModuleList(_DecoderLayer(d_model, nhead, dim_ff, dropout)
                                    for _ in range(n_layers))

    def forward(self, memory, memory_mask=None) -> dict:
        """memory (B, L, d) fused read tokens; memory_mask (B, L) True=valid. -> {gene: (B, d)}."""
        q = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)             # (B, 3, d)
        kpm = (~memory_mask) if memory_mask is not None else None                 # True = ignore (pad)
        for layer in self.layers:
            q = layer(q, memory, mem_key_padding_mask=kpm)
        return {g: q[:, i] for i, g in enumerate(GENES)}
