"""The shared per-position trunk: segmentation read from every position of one fully convolutional trunk.

``lab-trunk-v1`` keeps conv-v1's design (segment first, split, classify each gene, then metadata) and changes how it
segments. conv-v1 pools each gene's read through its own conv tower into one vector and maps it to the position
logits with a dense layer per boundary. Here one trunk (``DilatedTrunk``) keeps the read's length and gives every
position a feature vector, computed once per forward pass from the orientation-corrected embedding, and each gene's
start and end logits are a 1x1 convolution of it (``TrunkGeneBranch.boundary``). The logits are decoded as conv-v1
decodes them (the read-masked soft-argmax). The classifiers and the meta tower are conv-v1's, on the embedding:
``TrunkGeneBranch.classify`` and ``EmbeddingMetaTower`` take the trunk's features and do not read them yet.

Shapes: B = batch, L = ``cfg.max_seq_length``, E = ``cfg.embed_dim``, C = ``TRUNK_CHANNELS``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from alignair.core.components import ModelComponents, architecture_components
from alignair.core.config import AlignAIRConfig, GeneSpec
from alignair.core.gene_branch import GeneBranch, build_tower
from alignair.core.layers import SoftCutoutLayer
from alignair.errors import ConfigurationError

from . import register

TRUNK_CHANNELS = 128
# Two passes of doubling dilations; with kernel 5 each position sees 1 + 4 x 126 = 505 positions.
DILATIONS = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)


class _ResidualBlock(nn.Module):
    """``x + Dropout(GELU(Conv1d_dilated(LayerNorm(x) * keep)))`` on (B, L, C), keeping the length. LayerNorm
    normalizes each position over its channels, so no statistic is shared between positions or reads; ``keep``
    (B, L, 1) zeroes the normalized pad positions, where LayerNorm would leave its bias, before the convolution reads
    them."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding="same")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        h = self.conv((self.norm(x) * keep).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(h))


class DilatedTrunk(nn.Module):
    """``(emb (B, L, E), mask (B, L)) -> (B, L, channels)``: a 1x1 input projection, then pre-norm residual blocks of
    same-padded dilated convolutions. Pad positions (``mask`` False) are zeroed after the projection, after each
    block's normalization and after every block, so a convolution reads a pad position as it reads the zero padding
    past the window: a read's features depend neither on what its padding holds nor on how much padding there is.
    Each position sees ``receptive_field = 1 + (kernel - 1) * sum(dilations)`` positions around it, clipped to the
    read."""

    def __init__(self, in_channels: int, channels: int = TRUNK_CHANNELS, kernel: int = 5,
                 dilations: tuple[int, ...] = DILATIONS, dropout: float = 0.1):
        super().__init__()
        self.project = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList(_ResidualBlock(channels, kernel, dilation, dropout) for dilation in dilations)
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keep = mask.unsqueeze(-1).to(emb.dtype)
        x = self.project(emb.transpose(1, 2)).transpose(1, 2) * keep
        for block in self.blocks:
            x = block(x, keep) * keep
        return x


class TrunkGeneBranch(nn.Module):
    """One gene's branch on the trunk. ``segment`` reads its start and end logits (B, L) from the trunk's features
    with one 1x1 convolution (``boundary``) and decodes them as ``GeneBranch.segment`` does: the raw logits, and the
    soft-argmax expectation (B, 1) over the read's positions (the end over ``end_mask`` when given). ``classify`` is
    conv-v1's classifier, built with conv-v1's calls and computed by ``GeneBranch``'s own code on the embedding.

    For nucleotide models only: half-open boundaries (the protein models') are refused, since their end slot past a
    read's last residue is a pad position, whose trunk features are zero, so its end logit would be the boundary's
    bias for every read."""

    def __init__(self, spec: GeneSpec, cfg: AlignAIRConfig):
        super().__init__()
        if cfg.boundary_mode == "half_open":
            raise ConfigurationError(
                "lab-trunk-v1 is a nucleotide candidate and cannot segment half-open boundaries: the end slot past a "
                "read's last residue is a pad position, which the trunk zeroes", context={"field": "boundary_mode"},
                hint="Use residue boundaries (a nucleotide model).")
        self.classification_mode = cfg.classification_mode
        L = cfg.max_seq_length
        cls_out, cls_filters, latent = cfg.cls_spec(spec)
        self.boundary = nn.Conv1d(TRUNK_CHANNELS, 2, 1)                  # channel 0: start logits, 1: end logits
        self.cutout = SoftCutoutLayer(L, k=3.0)
        self.cls_tower = build_tower(cfg, spec.cls_kernels, filters=cls_filters, out=cls_out)
        self.cls_mid = nn.Linear(cls_out, latent)
        self.cls_head = nn.Linear(latent, spec.allele_count)
        self.register_buffer("_pos", torch.arange(L, dtype=torch.float32), persistent=False)

    # conv-v1's decoding and classification code, run on this branch's modules of the same names
    _soft_argmax = GeneBranch._soft_argmax
    classification_features = GeneBranch.classification_features

    def segment(self, emb: torch.Tensor, mask: torch.Tensor | None = None, end_mask: torch.Tensor | None = None, *,
                features: torch.Tensor):
        """features (B, L, C) -> (start_logits, end_logits, start_exp, end_exp); ``emb`` is not read."""
        s_log, e_log = self.boundary(features.transpose(1, 2)).unbind(1)
        s_dec, e_dec = s_log, e_log
        if mask is not None:
            neg = torch.finfo(s_log.dtype).min
            s_dec = s_log.masked_fill(~mask, neg)
            e_dec = e_log.masked_fill(~(mask if end_mask is None else end_mask), neg)
        return s_log, e_log, self._soft_argmax(s_dec), self._soft_argmax(e_dec)

    def classify(self, emb: torch.Tensor, s_exp: torch.Tensor, e_exp: torch.Tensor, *,
                 features: torch.Tensor) -> torch.Tensor:
        """conv-v1's allele scores (B, A) on the embedding soft-cut to the span; ``features`` is not read."""
        return GeneBranch.classify(self, emb, s_exp, e_exp)

    def component_modules(self) -> dict[str, nn.Module]:
        return {name: getattr(self, name) for name in ("boundary", "cutout", "cls_tower", "cls_mid", "cls_head")}


class EmbeddingMetaTower(nn.Module):
    """conv-v1's meta tower behind the trunk's call: ``forward(emb, features=...)`` is ``tower(emb)`` (B, block_out)."""

    def __init__(self, tower: nn.Module):
        super().__init__()
        self.tower = tower

    def forward(self, emb: torch.Tensor, *, features: torch.Tensor | None = None) -> torch.Tensor:
        return self.tower(emb)


def embedding_meta_tower(cfg: AlignAIRConfig) -> EmbeddingMetaTower:
    """The meta tower conv-v1 builds for ``cfg``, behind the trunk's call."""
    return EmbeddingMetaTower(architecture_components("conv-v1").meta_tower(cfg))


def build_trunk(cfg: AlignAIRConfig) -> DilatedTrunk:
    return DilatedTrunk(cfg.embed_dim, TRUNK_CHANNELS)


def build_trunk_heads(cfg: AlignAIRConfig) -> dict[str, nn.Module]:
    """The heads read from the trunk's features: none in lab-trunk-v1."""
    return {}


register("lab-trunk-v1", ModelComponents(trunk=build_trunk, gene_branch=TrunkGeneBranch,
                                         meta_tower=embedding_meta_tower, trunk_heads=build_trunk_heads))
