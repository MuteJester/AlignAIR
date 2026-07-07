"""Faithful PyTorch port of TF ``SingleChainAlignAIR`` (generalized single-chain aligner).

Heavy chain = ``has_d=True``; light chain = ``has_d=False`` (drops all D heads/towers). Architecture
(TF ``SingleChainAlignAIR.setup_model_layers``/``call``): token+position embedding -> shared conv
feature towers -> soft-argmax segmentation -> differentiable soft-cutout masking -> per-gene sigmoid
multi-label allele heads, plus mutation/indel/productivity heads off a shared ``meta`` tower.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.alignair_config import AlignAIRConfig
from .layers import ConvResidualFeatureExtractionBlock, SoftCutoutLayer, TokenAndPositionEmbedding

_SEG_KERNELS = [3, 3, 3, 2, 5]        # N=4 towers (meta + segmentation)
_V_J_CLS_KERNELS = [3, 3, 3, 2, 2, 2, 5]  # N=6 (V/J classification)
_D_CLS_KERNELS = [3, 3, 2, 2, 5]      # N=4 (D classification, shallower)


class SingleChainAlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        self.cfg = cfg
        self.genes = ["v", "j"] + (["d"] if cfg.has_d else [])
        L, C, F_ = cfg.max_seq_length, cfg.embed_dim, cfg.filters

        self.embedding = TokenAndPositionEmbedding(cfg.vocab_size, C, L)

        def block(N, kernels):
            return ConvResidualFeatureExtractionBlock(C, N=N, kernels=kernels, max_len=L,
                                                      filters=F_, out=cfg.block_out)

        self.meta_tower = block(4, _SEG_KERNELS)
        self.seg_towers = nn.ModuleDict({g: block(4, _SEG_KERNELS) for g in self.genes})
        cls_spec = {"v": (6, _V_J_CLS_KERNELS), "j": (6, _V_J_CLS_KERNELS), "d": (4, _D_CLS_KERNELS)}
        self.cls_towers = nn.ModuleDict({g: block(*cls_spec[g]) for g in self.genes})

        # segmentation heads: 576 -> L position logits, per boundary
        self.seg_heads = nn.ModuleDict(
            {f"{g}_{b}": nn.Linear(cfg.block_out, L) for g in self.genes for b in ("start", "end")})

        # classification heads: 576 -> latent (swish) -> count (sigmoid, multi-label)
        self.cls_mid = nn.ModuleDict({g: nn.Linear(cfg.block_out, cfg.latent(g)) for g in self.genes})
        counts = {"v": cfg.v_allele_count, "j": cfg.j_allele_count, "d": cfg.d_allele_count}
        self.cls_head = nn.ModuleDict({g: nn.Linear(cfg.latent(g), counts[g]) for g in self.genes})

        # analysis heads off meta
        self.mutation_rate_mid = nn.Linear(cfg.block_out, L)
        self.mutation_rate_head = nn.Linear(L, 1)
        self.indel_count_mid = nn.Linear(cfg.block_out, L)
        self.indel_count_head = nn.Linear(L, 1)
        self.productive_head = nn.Linear(cfg.block_out, 1)
        self.drop = nn.Dropout(0.05)

        self.cutout = SoftCutoutLayer(L, k=3.0)
        self.register_buffer("_positions", torch.arange(L, dtype=torch.float32), persistent=False)

    def _soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)                       # (B, L)
        return (probs * self._positions).sum(-1, keepdim=True)  # (B, 1)

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokenized_sequence"]
        emb = self.embedding(tokens)                            # (B, L, C)
        meta = self.meta_tower(emb)                             # (B, 576)
        out: dict = {}

        # segmentation + soft-argmax expectations
        exp = {}
        for g in self.genes:
            seg_feat = self.seg_towers[g](emb)
            for b in ("start", "end"):
                logits = self.seg_heads[f"{g}_{b}"](seg_feat)   # (B, L)
                out[f"{g}_{b}_logits"] = logits
                e = self._soft_argmax(logits)
                out[f"{g}_{b}"] = e
                exp[f"{g}_{b}"] = e

        # analysis heads (relu + kernel-clamped scale)
        out["mutation_rate"] = F.relu(self.mutation_rate_head(
            self.drop(F.gelu(self.mutation_rate_mid(meta)))))
        out["indel_count"] = F.relu(self.indel_count_head(
            self.drop(F.gelu(self.indel_count_mid(meta)))))
        out["productive"] = torch.sigmoid(self.productive_head(self.drop(meta)))

        # soft-cutout masking -> per-gene classification (multi-label sigmoid)
        for g in self.genes:
            mask = self.cutout(exp[f"{g}_start"], exp[f"{g}_end"]).unsqueeze(-1)  # (B, L, 1)
            feat = self.cls_towers[g](emb * mask)
            out[f"{g}_allele"] = torch.sigmoid(self.cls_head[g](F.silu(self.cls_mid[g](feat))))
        return out

    @torch.no_grad()
    def clamp_params(self) -> None:
        """TF MinMaxValueConstraint on the analysis-head kernels (applied after each optimizer step)."""
        self.mutation_rate_head.weight.clamp_(0.0, 1.0)
        self.indel_count_head.weight.clamp_(0.0, 50.0)
