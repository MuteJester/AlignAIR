"""AlignAIR model registry: resolve shipped model ids to local artifacts from configurable
registries (HuggingFace by default), with a verified local cache. Inference-safe (no pickle)."""
from __future__ import annotations

from . import sources
from .sources import DEFAULT_REGISTRY, OfflineError, find_model, load_registry, resolve_sources

__all__ = ["sources", "DEFAULT_REGISTRY", "OfflineError", "find_model", "load_registry",
           "resolve_sources"]
