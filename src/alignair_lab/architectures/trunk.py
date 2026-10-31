"""The shared per-position trunk: segmentation read from every position of one fully convolutional trunk.

``lab-trunk-v1`` keeps conv-v1's design (segment first, split, classify each gene, then metadata) and changes how it
segments. conv-v1 pools each gene's read through its own conv tower into one vector and maps it to the position
logits with a dense layer per boundary. Here one trunk (``DilatedTrunk``) keeps the read's length and gives every
position a feature vector, computed once per forward pass from the orientation-corrected embedding, and each gene's
start and end logits are a 1x1 convolution of it (``TrunkGeneBranch.boundary``). The logits are decoded as conv-v1
decodes them (the read-masked soft-argmax). The classifiers and the meta tower are conv-v1's, on the embedding:
``TrunkGeneBranch.classify`` and ``EmbeddingMetaTower`` take the trunk's features and do not read them yet.

``lab-trunk-v1`` segments residue boundaries: its end slot k is read from base k, and the slot past a read's last
base, a zeroed pad, is never supervised (the residue end mask leaves it out). ``lab-trunk-v2`` segments half-open
boundaries (the model option ``boundary_mode=half_open``): its end slot k is read from base k - 1, the segment's last
base, so a segment that runs to the read's end has a supervised end slot read from real features
(``TrunkGeneBranchV2``). Everything else is lab-trunk-v1's.

The model option ``region_head=true`` gives either architecture a region head (``RegionHead``, built by
``build_trunk_heads``): every position's recombination region (pre, V, N1, D, N2, J, post) as a 1x1 convolution of the
trunk's features, trained by the product's ``regions`` term. It is built after every other module, so the rest of the
model is as it is with the flag off.

The model option ``anchor_heads=true`` gives either architecture V and J anchor heads (``AnchorHead``): each read
position's logit of starting the conserved V Cys (or J Trp/Phe) codon, plus a null logit ("not observed"), trained by
the product's ``v_anchor``/``j_anchor`` terms. The product derives the junction, frame, stop codon and productivity
from the decoded anchors, so the learned productive head is not built (the modules built after the meta heads then
start from other random weights than with the flag off).

The model option ``d_presence_head=true`` gives either architecture a D presence head (``DPresenceHead``): absent, too
short to call or callable, from the read's pooled trunk features, trained by the product's ``d_presence`` term; the
product gates the D call on it. Built last, it leaves the rest of the model as it is with the flag off.

Shapes: B = batch, L = ``cfg.max_seq_length``, E = ``cfg.embed_dim``, C = ``TRUNK_CHANNELS``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from alignair.core.anchors import ANCHOR_GENES, ANCHOR_LOGITS
from alignair.core.d_presence import D_CLASSES, D_PRESENCE_LOGITS
from alignair.core.components import ModelComponents, architecture_components
from alignair.core.config import AlignAIRConfig, GeneSpec
from alignair.core.gene_branch import GeneBranch, build_tower
from alignair.core.layers import SoftCutoutLayer
from alignair.errors import ConfigurationError
from alignair.neural.heads.region import READ_REGIONS

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
        self._check_boundary_mode(cfg)
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

    @staticmethod
    def _check_boundary_mode(cfg: AlignAIRConfig) -> None:
        if cfg.boundary_mode == "half_open":
            raise ConfigurationError(
                "lab-trunk-v1 is a nucleotide candidate and cannot segment half-open boundaries: the end slot past a "
                "read's last residue is a pad position, which the trunk zeroes", context={"field": "boundary_mode"},
                hint="Use residue boundaries (a nucleotide model).")

    def _end_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """The end logits (B, L) from the boundary convolution's (B, 2, L): slot k's from the features of base k."""
        return logits[:, 1]

    def segment(self, emb: torch.Tensor, mask: torch.Tensor | None = None, end_mask: torch.Tensor | None = None, *,
                features: torch.Tensor):
        """features (B, L, C) -> (start_logits, end_logits, start_exp, end_exp); ``emb`` is not read."""
        logits = self.boundary(features.transpose(1, 2))
        s_log, e_log = logits[:, 0], self._end_logits(logits)
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


class TrunkGeneBranchV2(TrunkGeneBranch):
    """lab-trunk-v2's branch: lab-trunk-v1's, for half-open boundaries. Half-open end slot k ends a segment whose last
    base is k - 1, so its logit is the end head on the features of base k - 1. The slot after a read's last base,
    where a segment running to the read's end stops (J in nearly every full-length read), so reads that base's
    features, not a zeroed pad's. Slot 0 ends no segment (a half-open model never allows it) and holds 0. The start
    logits, the decoding, the classifier and the parameters are lab-trunk-v1's.

    Residue boundaries are refused: their end slot is a segment's last base itself, and their slot past the read's
    last base is never supervised."""

    @staticmethod
    def _check_boundary_mode(cfg: AlignAIRConfig) -> None:
        if cfg.boundary_mode != "half_open":
            raise ConfigurationError(
                f"lab-trunk-v2 segments half-open boundaries only, not {cfg.boundary_mode!r} ones: it reads end slot "
                "k from the features of base k - 1, the last base of a half-open segment, while a residue end slot "
                "is that base itself", context={"field": "boundary_mode"},
                hint="Set the model option boundary_mode=half_open, or use lab-trunk-v1 for residue boundaries.")

    def _end_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Slot k's from the features of base k - 1; slot 0's is 0."""
        return F.pad(logits[:, 1, :-1], (1, 0))


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


class RegionHead(nn.Module):
    """The recombination region of every position: ``(features (B, L, C), mask) -> logits (B, L, 7)`` over
    ``READ_REGIONS`` (pre, V, N1, D, N2, J, post), one 1x1 convolution of the trunk's features. A pad position's
    logits are the bias (its features are zero); the region term reads the read's positions only."""

    def __init__(self, channels: int = TRUNK_CHANNELS):
        super().__init__()
        self.conv = nn.Conv1d(channels, len(READ_REGIONS), 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.conv(features.transpose(1, 2)).transpose(1, 2)


class AnchorHead(nn.Module):
    """One conserved anchor's position, or that the read does not show it: ``(features (B, L, C), mask (B, L)) ->
    logits (B, L + 1)``. Slot k < L, the anchor codon starting at base k, is a 1x1 convolution of position k's
    features; slot L, the null slot ("not observed"), is a linear map of the read's mean and max features over its
    positions (``mask``). A pad position's slot is the convolution's bias (its features are zero); the anchor term
    and the decoder read the read's positions and the null slot only."""

    def __init__(self, channels: int = TRUNK_CHANNELS):
        super().__init__()
        self.position = nn.Conv1d(channels, 1, 1)
        self.null = nn.Linear(2 * channels, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = self.position(features.transpose(1, 2))[:, 0]
        return torch.cat([positions, self.null(_pooled(features, mask))], -1)


def _pooled(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The read's mean and max features over its positions (``mask``): (B, L, C) -> (B, 2C); 0 for a read with no
    position."""
    keep = mask.unsqueeze(-1)
    mean = (features * keep).sum(1) / keep.sum(1).clamp_min(1)
    peak = features.masked_fill(~keep, float("-inf")).amax(1)
    peak = torch.where(mask.any(1, keepdim=True), peak, torch.zeros_like(peak))     # a read with no position
    return torch.cat([mean, peak], -1)


class DPresenceHead(nn.Module):
    """Whether the read shows a D, and one long enough to call: ``(features (B, L, C), mask (B, L)) -> logits (B, 3)``
    over ``D_CLASSES`` (absent, too short to call, callable), from the read's mean and max trunk features through
    Linear(2C, 64), GELU and Linear(64, 3). The product gates the D call on its argmax."""

    def __init__(self, channels: int = TRUNK_CHANNELS, hidden: int = 64):
        super().__init__()
        self.hidden = nn.Linear(2 * channels, hidden)
        self.classes = nn.Linear(hidden, len(D_CLASSES))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.classes(F.gelu(self.hidden(_pooled(features, mask))))


def build_trunk_heads(cfg: AlignAIRConfig) -> dict[str, nn.Module]:
    """The heads read from the trunk's features, each built only when its flag is on: ``region_logits``
    (``RegionHead``) with ``region_head``; ``v_anchor_logits`` and ``j_anchor_logits`` (``AnchorHead``: the V Cys
    and J Trp/Phe codon starts) with ``anchor_heads``; ``d_presence_logits`` (``DPresenceHead``) with
    ``d_presence_head``. With every flag off, lab-trunk-v1 and lab-trunk-v2 have none."""
    heads = {}
    if cfg.region_head:
        heads["region_logits"] = RegionHead()
    if cfg.anchor_heads:
        heads.update({ANCHOR_LOGITS[gene]: AnchorHead() for gene in ANCHOR_GENES})       # V first, then J
    if cfg.d_presence_head:
        heads[D_PRESENCE_LOGITS] = DPresenceHead()
    return heads


register("lab-trunk-v1", ModelComponents(trunk=build_trunk, gene_branch=TrunkGeneBranch,
                                         meta_tower=embedding_meta_tower, trunk_heads=build_trunk_heads))
register("lab-trunk-v2", ModelComponents(trunk=build_trunk, gene_branch=TrunkGeneBranchV2,
                                         meta_tower=embedding_meta_tower, trunk_heads=build_trunk_heads))
