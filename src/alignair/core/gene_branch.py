"""The two composable building blocks the AlignAIR model is assembled from.

``GeneBranch`` is one gene's entire pipeline in a single cohesive, testable unit (segmentation tower
-> boundary logits -> soft-argmax -> soft-cutout mask -> classification tower -> allele head). The
model just holds one per gene. ``MetaHead`` is one meta-tower-derived prediction (mutation / indel /
productivity / chain_type) with an optional hidden layer, output activation and post-step weight
clamp. Together they replace the flat, gene-branched heads that were previously spread across the
model's ``__init__`` and ``forward``.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AlignAIRConfig, GeneSpec
from .layers import ConvResidualFeatureExtractionBlock, SoftCutoutLayer


def build_tower(cfg: AlignAIRConfig, kernels: tuple) -> ConvResidualFeatureExtractionBlock:
    """A residual conv feature tower sized from a kernel schedule (N = len(kernels) - 1)."""
    return ConvResidualFeatureExtractionBlock(
        cfg.embed_dim, N=len(kernels) - 1, kernels=list(kernels), max_len=cfg.max_seq_length,
        filters=cfg.filters, out=cfg.block_out)


class GeneBranch(nn.Module):
    """One gene's full V(D)J pipeline. ``segment`` predicts boundary logits + soft-argmax positional
    expectations; ``classify`` soft-cuts the embedding to the predicted span and calls the allele
    multi-label head. Faithful to the per-gene ops previously smeared across ``seg_towers`` /
    ``seg_heads`` / ``cls_towers`` / ``cls_mid`` / ``cls_head``."""

    def __init__(self, spec: GeneSpec, cfg: AlignAIRConfig):
        super().__init__()
        L = cfg.max_seq_length
        latent = spec.latent(cfg.latent_size_factor)
        self.seg_tower = build_tower(cfg, spec.seg_kernels)
        self.start_head = nn.Linear(cfg.block_out, L)
        self.end_head = nn.Linear(cfg.block_out, L)
        self.cutout = SoftCutoutLayer(L, k=3.0)
        self.cls_tower = build_tower(cfg, spec.cls_kernels)
        self.cls_mid = nn.Linear(cfg.block_out, latent)
        self.cls_head = nn.Linear(latent, spec.allele_count)
        self.register_buffer("_pos", torch.arange(L, dtype=torch.float32), persistent=False)

    def _soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) * self._pos).sum(-1, keepdim=True)   # (B, 1)

    def segment(self, emb: torch.Tensor, mask: torch.Tensor | None = None):
        """emb (B, L, C) -> (start_logits, end_logits, start_exp, end_exp).

        ``mask`` (B, L bool, True = valid read position) restricts the soft-argmax to the read so the
        expectation cannot be pulled into the right-padding. Raw logits are returned unmasked (the
        loss re-applies the mask); only the decoded expectation is padding-masked."""
        f = self.seg_tower(emb)
        s_log, e_log = self.start_head(f), self.end_head(f)
        s_dec, e_dec = s_log, e_log
        if mask is not None:
            neg = torch.finfo(s_log.dtype).min
            s_dec = s_log.masked_fill(~mask, neg)
            e_dec = e_log.masked_fill(~mask, neg)
        return s_log, e_log, self._soft_argmax(s_dec), self._soft_argmax(e_dec)

    def classify(self, emb: torch.Tensor, s_exp: torch.Tensor, e_exp: torch.Tensor) -> torch.Tensor:
        mask = self.cutout(s_exp, e_exp).unsqueeze(-1)                          # (B, L, 1)
        f = self.cls_tower(emb * mask)
        return torch.sigmoid(self.cls_head(F.silu(self.cls_mid(f))))           # multi-label allele probs


class MetaHead(nn.Module):
    """A prediction off the shared meta tower: optional (Linear + act + dropout) hidden layer, then a
    Linear head with an optional output activation and an optional post-optimizer-step weight clamp
    (the TF ``MinMaxValueConstraint``). Covers mutation-rate / indel-count / productivity / chain_type
    with the same tiny module instead of four bespoke head pairs."""

    def __init__(self, in_dim: int, out_dim: int, *, mid_dim: Optional[int] = None,
                 act: Callable = F.gelu, out_act: Optional[Callable] = None,
                 dropout: float = 0.05, clamp: Optional[tuple] = None):
        super().__init__()
        self.mid = nn.Linear(in_dim, mid_dim) if mid_dim else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(mid_dim if mid_dim else in_dim, out_dim)
        self._act = act
        self._out_act = out_act
        self._clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self._act(self.mid(x))) if self.mid is not None else self.drop(x)
        y = self.head(x)
        return self._out_act(y) if self._out_act is not None else y

    @torch.no_grad()
    def clamp_(self) -> None:
        if self._clamp is not None:
            self.head.weight.clamp_(*self._clamp)
