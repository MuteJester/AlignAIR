"""AlignAIR model registry: resolve shipped model ids to local artifacts from configurable
registries (HuggingFace by default), with a verified local cache. Inference-safe (no pickle)."""
from __future__ import annotations

from . import cache, hf, sources, updates
from .cache import DownloadIntegrityError, resolve_model
from .hf import download_from_hub, is_hf_repo_spec, parse_hf_spec
from .publish import publish_bundle, publish_local, upload_version
from .sources import DEFAULT_REGISTRY, OfflineError, find_model, load_registry, resolve_sources
from .updates import maybe_notify_updates
from .validate import validate_registry

__all__ = [
    "DEFAULT_REGISTRY",
    "DownloadIntegrityError",
    "OfflineError",
    "cache",
    "download_from_hub",
    "find_model",
    "hf",
    "is_hf_repo_spec",
    "load_registry",
    "maybe_notify_updates",
    "parse_hf_spec",
    "publish_bundle",
    "publish_local",
    "resolve_model",
    "resolve_sources",
    "sources",
    "updates",
    "upload_version",
    "validate_registry",
]
