"""AlignAIR model registry: resolve shipped model ids to local artifacts from configurable
registries (HuggingFace by default), with a verified local cache. Inference-safe (no pickle).

A registry list is passed as ``registry=`` (one registry may be given as a string). The maintainer's publishing tools
are in ``alignair.registry.publish``."""
from __future__ import annotations

from .cache import resolve_model
from .hf import download_from_hub, is_hf_repo_spec, parse_hf_spec
from .sources import DEFAULT_REGISTRY, find_model, load_registry, resolve_sources
from .updates import maybe_notify_updates
from .validate import validate_registry

__all__ = [
    "DEFAULT_REGISTRY",
    "download_from_hub",
    "find_model",
    "is_hf_repo_spec",
    "load_registry",
    "maybe_notify_updates",
    "parse_hf_spec",
    "resolve_model",
    "resolve_sources",
    "validate_registry",
]
