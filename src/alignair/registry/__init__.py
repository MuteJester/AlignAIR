"""AlignAIR model registry: resolve shipped model ids to local artifacts from configurable
registries (HuggingFace by default), with a verified local cache. Inference-safe (no pickle)."""
from __future__ import annotations

from . import cache, hf, sources, updates
from .cache import IntegrityError, resolve_model
from .hf import download_from_hub, is_hf_repo_spec, parse_hf_spec
from .publish import publish_local
from .sources import DEFAULT_REGISTRY, OfflineError, find_model, load_registry, resolve_sources
from .updates import maybe_notify_updates
from .validate import validate_registry

__all__ = ["sources", "cache", "hf", "updates", "DEFAULT_REGISTRY", "OfflineError", "IntegrityError",
           "find_model", "load_registry", "resolve_sources", "resolve_model", "maybe_notify_updates",
           "publish_local", "validate_registry", "is_hf_repo_spec", "parse_hf_spec", "download_from_hub"]
