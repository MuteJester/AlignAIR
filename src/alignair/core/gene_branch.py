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
from ..errors import ConfigurationError


def build_tower(cfg: AlignAIRConfig, kernels: tuple, filters: int | None = None,
                out: int | None = None) -> ConvResidualFeatureExtractionBlock:
    """A residual conv feature tower sized from a kernel schedule (N = len(kernels) - 1). ``filters`` /
    ``out`` override the config defaults (used by the auto-scaled per-gene classification tower)."""
    return ConvResidualFeatureExtractionBlock(
        cfg.embed_dim, N=len(kernels) - 1, kernels=list(kernels), max_len=cfg.max_seq_length,
        filters=cfg.filters if filters is None else filters,
        out=cfg.block_out if out is None else out)


class GeneBranch(nn.Module):
    """One gene's full V(D)J pipeline. ``segment`` predicts boundary logits + soft-argmax positional
    expectations; ``classify`` soft-cuts the embedding to the predicted span and calls the allele
    multi-label head. Faithful to the per-gene ops previously smeared across ``seg_towers`` /
    ``seg_heads`` / ``cls_towers`` / ``cls_mid`` / ``cls_head``."""

    def __init__(self, spec: GeneSpec, cfg: AlignAIRConfig, *,
                 segmentation_tower: nn.Module | None = None,
                 classification_tower: nn.Module | None = None):
        super().__init__()
        for name, tower in (("segmentation_tower", segmentation_tower), ("classification_tower", classification_tower)):
            if tower is not None and not isinstance(tower, nn.Module):
                raise ConfigurationError("replacement tower must be a torch module", context={"field": name})
        self.classification_mode = cfg.classification_mode
        L = cfg.max_seq_length
        # allele-discrimination sizing: legacy fixed dims, or auto-scaled to this gene's allele count
        cls_out, cls_filters, latent = cfg.cls_spec(spec)
        self.seg_tower = (build_tower(cfg, spec.seg_kernels)
                          if segmentation_tower is None else segmentation_tower)
        self.start_head = nn.Linear(cfg.block_out, L)
        self.end_head = nn.Linear(cfg.block_out, L)
        self.cutout = SoftCutoutLayer(L, k=3.0)
        self.cls_tower = (build_tower(cfg, spec.cls_kernels, filters=cls_filters, out=cls_out)
                          if classification_tower is None else classification_tower)
        self.cls_mid = nn.Linear(cls_out, latent)
        self.cls_head = nn.Linear(latent, spec.allele_count)
        self.register_buffer("_pos", torch.arange(L, dtype=torch.float32), persistent=False)

    def _soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) * self._pos).sum(-1, keepdim=True)   # (B, 1)

    def segment(self, emb: torch.Tensor, mask: torch.Tensor | None = None,
                end_mask: torch.Tensor | None = None):
        """emb (B, L, C) -> (start_logits, end_logits, start_exp, end_exp).

        ``mask`` (B, L bool, True = valid read position) restricts the soft-argmax to the read so the
        expectation cannot be pulled into the right-padding. Raw logits are returned unmasked (the
        loss re-applies the mask); only the decoded expectation is padding-masked. An optional
        ``end_mask`` selects exclusive-end slots independently of the start positions."""
        f = self.seg_tower(emb)
        s_log, e_log = self.start_head(f), self.end_head(f)
        s_dec, e_dec = s_log, e_log
        if mask is not None:
            neg = torch.finfo(s_log.dtype).min
            s_dec = s_log.masked_fill(~mask, neg)
            e_dec = e_log.masked_fill(~(mask if end_mask is None else end_mask), neg)
        return s_log, e_log, self._soft_argmax(s_dec), self._soft_argmax(e_dec)

    def segment_with_masks(self, emb: torch.Tensor, start_mask: torch.Tensor, end_mask: torch.Tensor):
        """Explicit start/end-mask form of segment, shared by protein and nucleotide branches."""
        return self.segment(emb, start_mask, end_mask)

    def classification_features(self, emb: torch.Tensor, s_exp: torch.Tensor,
                                e_exp: torch.Tensor) -> torch.Tensor:
        """Penultimate representation consumed by cls_head, before score activation."""
        mask = self.cutout(s_exp, e_exp).unsqueeze(-1)                          # (B, L, 1)
        f = self.cls_tower(emb * mask)
        return F.silu(self.cls_mid(f))

    def classify(self, emb: torch.Tensor, s_exp: torch.Tensor, e_exp: torch.Tensor) -> torch.Tensor:
        logits = self.cls_head(self.classification_features(emb, s_exp, e_exp))
        return torch.sigmoid(logits) if self.classification_mode == "multilabel" else logits

    def component_modules(self) -> dict[str, nn.Module]:
        """Named inspection points; custom branches can extend this contract."""
        names = ("seg_tower", "start_head", "end_head", "cutout", "cls_tower", "cls_mid", "cls_head")
        return {name: getattr(self, name) for name in names}


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
