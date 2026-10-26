"""Stage-3 Task 3.2 — ApplicabilityEncoder: a pre-norm transformer with rotary positions and
padding-aware attention. Emits a CLS vector and a masked-mean vector. Padding-invariant: appending pad
tokens (mask=False) does not change the outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignair.aa.guard.tokenizer import VOCAB_SIZE


def _rope_tables(seq_len, head_dim, device, dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # [L, head_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x, cos, sin):
    # x: [B, H, L, D]; cos/sin: [L, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class _Block(nn.Module):
    def __init__(self, width, heads, ff, dropout):
        super().__init__()
        self.heads, self.hd = heads, width // heads
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding, cos, sin):
        B, L, W = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.heads, self.hd).transpose(1, 2)      # [B,H,L,D]
        k = k.view(B, L, self.heads, self.hd).transpose(1, 2)
        v = v.view(B, L, self.heads, self.hd).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        attn_mask = None
        if key_padding is not None:                                # [B,L] True where PAD
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(key_padding[:, None, None, :], float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, L, W)
        x = x + self.drop(self.proj(o))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class ApplicabilityEncoder(nn.Module):
    def __init__(self, width=128, depth=4, heads=4, ff=512, dropout=0.10, vocab_size=VOCAB_SIZE):
        super().__init__()
        assert width % heads == 0
        self.width = width
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([_Block(width, heads, ff, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, ids, mask):
        """ids[B,L], mask[B,L] (True=real residue/CLS). Returns (cls_vec[B,width], mean_vec[B,width])."""
        if ids.dim() == 1:
            ids, mask = ids[None], mask[None]
        x = self.embed(ids)
        key_padding = ~mask                                        # True where pad
        cos, sin = _rope_tables(x.shape[1], self.blocks[0].hd, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, key_padding, cos, sin)
        x = self.norm(x)
        cls_vec = x[:, 0]                                          # CLS at position 0
        m = mask.unsqueeze(-1).to(x.dtype)
        mean_vec = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # masked mean over valid positions
        return cls_vec, mean_vec
