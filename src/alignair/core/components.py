"""Explicit, application-owned construction of model components.

Register a new architecture name when replacing components. The name is stored in the model config;
its factories must be registered in a loading process too. Missing factories fail before weights load.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, fields
from collections.abc import Mapping
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..errors import ConfigurationError, error_context
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
    """The metadata heads on the meta tower's vector. With ``anchor_heads`` there is no productive head: productivity
    is derived from the predicted anchors (``predict.junction``)."""
    length = cfg.max_seq_length
    heads = {
        "mutation_rate": MetaHead(cfg.block_out, 1, mid_dim=length, out_act=F.relu, clamp=(0.0, 1.0)),
        "indel_count": MetaHead(cfg.block_out, 1, mid_dim=length, out_act=F.relu, clamp=(0.0, 50.0)),
    }
    if not getattr(cfg, "anchor_heads", False):
        heads["productive"] = MetaHead(cfg.block_out, 1, out_act=torch.sigmoid)
    if cfg.num_chain_types > 1:
        heads["chain_type_logits"] = MetaHead(cfg.block_out, cfg.num_chain_types, mid_dim=length)
    return heads


def no_trunk_heads(cfg: AlignAIRConfig) -> dict[str, nn.Module]:
    return {}


@dataclass(frozen=True)
class ModelComponents:
    """Factories for one architecture, independent of its alphabet and enabled tasks.

    Embeddings emit (batch, window, embed_dim). Gene branches implement segment, classify,
    and component_modules. Their boundary tensors are (batch, window)/(batch, 1), and class
    scores are (batch, labels). Auxiliary towers emit (batch, block_out).

    An optional trunk maps (embedding, mask) to shared per-position features (batch, window, channels),
    computed once per forward after orientation correction. When present, branches, the meta tower and
    the trunk heads receive them (branches and the tower as ``features=``; a head is called
    ``head(features, mask)`` and its result is written to the output under the head's name).

    An optional core replaces everything after orientation correction: the model then builds only its embedding,
    orientation head and core, and calls ``core(emb, mask, end_mask, *, tokens, batch)`` with the corrected
    embedding, the read and end masks, the corrected tokens and the raw input batch; the core's outputs are written to
    the model's output. A core that needs the reference defines ``bind_reference(reference)``; the model's builders
    and loaders call it. An optional objective (``objective(config) -> TrainingTasks``) is the architecture's training
    objective, which ``tasks.build_tasks`` returns in place of the built-in one.

    ``reference_agnostic``: the architecture's alleles are an input, not weights (its core reads them from the bound
    reference), so it takes any reference with its genes, alleles it never trained on included, and its calls follow
    the bound reference's alleles; a fixed-head architecture (the default) is tied to the reference it trained on.
    """
    embedding: Callable[[AlignAIRConfig], nn.Module] = _embedding
    gene_branch: Callable[[GeneSpec, AlignAIRConfig], GeneBranch] = GeneBranch
    orientation_head: Callable[[AlignAIRConfig], nn.Module] = _orientation
    meta_tower: Callable[[AlignAIRConfig], nn.Module] = _meta_tower
    meta_heads: Callable[[AlignAIRConfig], dict[str, nn.Module]] = build_meta_heads
    state_head: Callable[[AlignAIRConfig], nn.Module] = _state
    trunk: Callable[[AlignAIRConfig], nn.Module] | None = None
    trunk_heads: Callable[[AlignAIRConfig], Mapping[str, nn.Module]] = no_trunk_heads
    core: Callable[[AlignAIRConfig], nn.Module] | None = None
    objective: Callable[[AlignAIRConfig], object] | None = None
    reference_agnostic: bool = False

    def __post_init__(self):
        if type(self.reference_agnostic) is not bool:
            raise ConfigurationError("reference_agnostic must be a boolean", context={"component": "reference_agnostic"})
        for field in fields(self):
            if field.name == "reference_agnostic":
                continue
            if field.name in ("trunk", "core", "objective") and getattr(self, field.name) is None:
                continue
            if not callable(getattr(self, field.name)):
                raise ConfigurationError(f"component factory {field.name!r} must be callable",
                                         context={"component": field.name})


_ARCHITECTURES = {"conv-v1": ModelComponents()}
# The environment variable naming the modules (comma-separated) that register architectures defined outside the package.
ARCHITECTURE_MODULES = "ALIGNAIR_ARCHITECTURE_MODULES"


def _build_component(factory, *arguments, component):
    """Validate a factory result at its construction boundary without running a dummy forward."""
    with error_context(stage="model_construction", component=component):
        module = factory(*arguments)
        if not isinstance(module, nn.Module):
            raise ConfigurationError("component factory must return a torch module",
                                     context={"returned_type": type(module).__name__})
        return module


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
    """The registered components of ``name``. An architecture registered outside the package is found by importing
    the modules ``ALIGNAIR_ARCHITECTURE_MODULES`` names (comma-separated), which register it: a process that loads such
    a model, a subprocess included, takes them from its environment."""
    if isinstance(name, str) and name not in _ARCHITECTURES:
        for module in filter(None, (part.strip() for part in os.environ.get(ARCHITECTURE_MODULES, "").split(","))):
            try:
                importlib.import_module(module)
            except ImportError as exc:
                raise ConfigurationError(f"{ARCHITECTURE_MODULES} names {module!r}, which cannot be imported",
                                         context={"module": module}) from exc
    try:
        return _ARCHITECTURES[name]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"unknown model architecture: {name!r}",
                                 hint=f"Register its ModelComponents before constructing or loading the model, or name "
                                      f"the module that registers it in {ARCHITECTURE_MODULES}.") from exc
