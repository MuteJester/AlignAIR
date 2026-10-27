"""Explicit, application-owned construction of model components.

Register a new architecture name when replacing components. The name is stored in the model config;
its factories must be registered in a loading process too. Missing factories fail before weights load.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..errors import ConfigurationError
from ..neural.heads.state import PerPositionStateBranch
from .config import AlignAIRConfig, GeneSpec, _SEG_KERNELS
from .gene_branch import GeneBranch, MetaHead, build_tower
from .layers import EmbeddingOrientationHead, TokenAndPositionEmbedding


def _embedding(cfg):
    return TokenAndPositionEmbedding(cfg.vocab_size, cfg.embed_dim, cfg.max_seq_length)


def _orientation(cfg):
    return EmbeddingOrientationHead(cfg.embed_dim)


def _meta_tower(cfg):
    return build_tower(cfg, _SEG_KERNELS)


def _state(cfg):
    return PerPositionStateBranch(cfg.embed_dim)


def build_meta_heads(cfg: AlignAIRConfig) -> dict[str, nn.Module]:
    length = cfg.max_seq_length
    heads = {
        "mutation_rate": MetaHead(cfg.block_out, 1, mid_dim=length, out_act=F.relu, clamp=(0.0, 1.0)),
        "indel_count": MetaHead(cfg.block_out, 1, mid_dim=length, out_act=F.relu, clamp=(0.0, 50.0)),
        "productive": MetaHead(cfg.block_out, 1, out_act=torch.sigmoid),
    }
    if cfg.num_chain_types > 1:
        heads["chain_type_logits"] = MetaHead(cfg.block_out, cfg.num_chain_types, mid_dim=length)
    return heads


@dataclass(frozen=True)
class ModelComponents:
    """Factories for one architecture, independent of its alphabet and enabled tasks.

    Embeddings emit (batch, window, embed_dim). Gene branches implement segment, classify,
    and component_modules. Their boundary tensors are (batch, window)/(batch, 1), and class
    scores are (batch, labels). Auxiliary towers emit (batch, block_out).
    """
    embedding: Callable[[AlignAIRConfig], nn.Module] = _embedding
    gene_branch: Callable[[GeneSpec, AlignAIRConfig], GeneBranch] = GeneBranch
    orientation_head: Callable[[AlignAIRConfig], nn.Module] = _orientation
    meta_tower: Callable[[AlignAIRConfig], nn.Module] = _meta_tower
    meta_heads: Callable[[AlignAIRConfig], dict[str, nn.Module]] = build_meta_heads
    state_head: Callable[[AlignAIRConfig], nn.Module] = _state

    def __post_init__(self):
        for field in fields(self):
            if not callable(getattr(self, field.name)):
                raise ConfigurationError(f"component factory {field.name!r} must be callable",
                                         context={"component": field.name})


_ARCHITECTURES = {"conv-v1": ModelComponents()}


def register_architecture(name: str, components: ModelComponents) -> None:
    """Register a uniquely named architecture; existing definitions cannot be overwritten."""
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ConfigurationError("architecture name must be a non-empty string without surrounding whitespace")
    if not isinstance(components, ModelComponents):
        raise ConfigurationError("architecture components must be ModelComponents")
    if name in _ARCHITECTURES:
        raise ConfigurationError(f"architecture {name!r} is already registered")
    _ARCHITECTURES[name] = components


def architecture_components(name: str) -> ModelComponents:
    try:
        return _ARCHITECTURES[name]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"unknown model architecture: {name!r}",
                                 hint="Register its ModelComponents before constructing or loading the model.") from exc
