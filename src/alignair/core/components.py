"""Explicit, application-owned construction of model components.

Register a new architecture name when replacing components. The name is stored in the model config;
its factories must be registered in a loading process too. Missing factories fail before weights load.
"""
from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from torch import nn

from ..errors import ConfigurationError, error_context
from .config import AlignAIRConfig

if TYPE_CHECKING:
    from ..tasks.objective import TrainingTasks


def _embedding(cfg):
    from .embedding import TokenAndPositionEmbedding
    return TokenAndPositionEmbedding(cfg.vocab_size, cfg.embed_dim, cfg.max_seq_length)


def _orientation(cfg):
    from .orientation import EmbeddingOrientationHead
    return EmbeddingOrientationHead(cfg.embed_dim)


@dataclass(frozen=True)
class ModelComponents:
    """Factories for one architecture.

    The embedding emits (batch, window, embed_dim); the orientation head reads it with the read mask and gives the
    4-class orientation logits. The core does everything after orientation correction: the model calls
    ``core(emb, mask, end_mask, *, tokens, batch)`` with the corrected embedding, the read and end masks, the corrected
    tokens and the raw input batch, and writes the core's outputs to its own. A core that needs the reference defines
    ``bind_reference(reference)``; the model's builders and loaders call it. ``objective(config) -> TrainingTasks`` is
    the architecture's training objective (``tasks.build_tasks``).

    An architecture's alleles are an input, not weights: its core reads them from the bound reference, so a model
    takes any reference with its genes, alleles it never trained on included, and its calls are that reference's.
    """
    core: Callable[[AlignAIRConfig], nn.Module]
    objective: Callable[[AlignAIRConfig], TrainingTasks]
    embedding: Callable[[AlignAIRConfig], nn.Module] = _embedding
    orientation_head: Callable[[AlignAIRConfig], nn.Module] = _orientation

    def __post_init__(self):
        for field in fields(self):
            if not callable(getattr(self, field.name)):
                raise ConfigurationError(f"component factory {field.name!r} must be callable",
                                         context={"component": field.name})


def _chip_core(cfg: AlignAIRConfig) -> nn.Module:
    """The chip's core (``alignair.network.chip``) on a trunk ``cfg.trunk_channels`` wide; imported when built, so the
    registry does not import the network."""
    from ..network.chip import ChipCore
    return ChipCore(cfg)


def _chip_objective(cfg: AlignAIRConfig):
    """The chip's training objective (``alignair.network.objective``)."""
    from ..network.objective import ChipObjective
    return ChipObjective(cfg)


CHIP = ModelComponents(core=_chip_core, objective=_chip_objective)
_ARCHITECTURES = {"chip-v1": CHIP}
# The environment variable naming the modules (comma-separated) that register architectures defined outside the package.
ARCHITECTURE_MODULES = "ALIGNAIR_ARCHITECTURE_MODULES"


def build_component(factory, *arguments, component):
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
