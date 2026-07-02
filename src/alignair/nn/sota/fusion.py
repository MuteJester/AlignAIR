"""Bidirectional read <-> reference cross-attention fusion.

A clean 1-D re-implementation of GLIP's `BiAttentionBlock` / `BiMultiHeadAttention` deep
vision-language fusion (Microsoft GLIP, MIT), here applied to DNA: read tokens and candidate
germline embeddings attend to each other so the read representation becomes *conditioned on the
candidate set* (and vice-versa). A zero-initialised gate (Flamingo-style) makes the block a
no-op at initialisation, so it can be added to a converged encoder without disrupting it and is
learned in gradually. See sota/ATTRIBUTION.md.
"""
import torch
import torch.nn as nn


class BiCrossAttention(nn.Module):
    """Multi-head cross-attention: read attends candidates (and, if `bidirectional`, candidates
    attend read). Set `bidirectional=False` for read-only conditioning — when the caller discards
    the fused candidates, the candidate-direction projections would be dead weight."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, bidirectional: bool = True):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead, self.hd = nhead, d_model // nhead
        self.scale = self.hd ** -0.5
        self.bidirectional = bidirectional
        self.q_r = nn.Linear(d_model, d_model)                      # read attends candidates
        self.k_c = nn.Linear(d_model, d_model); self.v_c = nn.Linear(d_model, d_model)
        self.o_r = nn.Linear(d_model, d_model)
        if bidirectional:                                          # candidates attend read
            self.k_r = nn.Linear(d_model, d_model); self.v_r = nn.Linear(d_model, d_model)
            self.q_c = nn.Linear(d_model, d_model); self.o_c = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _heads(self, x):                                    # (B,L,d) -> (B,H,L,hd)
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.hd).transpose(1, 2)

    def _merge(self, x, out):                               # (B,H,L,hd) -> (B,L,d)
        B, H, L, hd = x.shape
        return out(x.transpose(1, 2).reshape(B, L, H * hd))

    def _attend(self, q, k, v, key_mask):
        a = (q @ k.transpose(-2, -1)) * self.scale          # (B,H,Lq,Lk)
        if key_mask is not None:
            a = a.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        return self.drop(a.softmax(dim=-1)) @ v

    def forward(self, read, cand, read_mask=None, cand_mask=None):
        qr = self._heads(self.q_r(read))
        kc, vc = self._heads(self.k_c(cand)), self._heads(self.v_c(cand))
        r_upd = self._merge(self._attend(qr, kc, vc, cand_mask), self.o_r)   # read attends candidates
        if not self.bidirectional:
            return r_upd, None
        qc = self._heads(self.q_c(cand))
        kr, vr = self._heads(self.k_r(read)), self._heads(self.v_r(read))
        c_upd = self._merge(self._attend(qc, kr, vr, read_mask), self.o_c)   # candidates attend read
        return r_upd, c_upd


class BiAttentionBlock(nn.Module):
    """Residual bidirectional fusion block. The tanh gate is zero-initialised, so at init the
    block returns its inputs unchanged (identity) and learns to fuse gradually."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.attn = BiCrossAttention(d_model, nhead, dropout, bidirectional)
        self.gate_r = nn.Parameter(torch.zeros(1))
        if bidirectional:
            self.gate_c = nn.Parameter(torch.zeros(1))

    def forward(self, read, cand, read_mask=None, cand_mask=None):
        dr, dc = self.attn(self.norm_r(read), self.norm_c(cand), read_mask, cand_mask)
        read = read + self.gate_r.tanh() * dr
        if self.bidirectional:
            cand = cand + self.gate_c.tanh() * dc
        return read, cand                                 # cand unchanged when read-only


class ReferenceFusion(nn.Module):
    """A stack of `n_layers` bidirectional fusion blocks (GLIP deep fusion)."""

    def __init__(self, d_model: int, nhead: int, n_layers: int = 2, dropout: float = 0.0,
                 bidirectional: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            BiAttentionBlock(d_model, nhead, dropout, bidirectional) for _ in range(n_layers))

    def forward(self, read, cand, read_mask=None, cand_mask=None):
        for blk in self.blocks:
            read, cand = blk(read, cand, read_mask, cand_mask)
        return read, cand
