"""HuggingFace Hub integration for downloading pretrained AlignAIR model bundles."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
HF_REPO_ID: str = "AlignAIR/AlignAIR-pretrained"

AVAILABLE_MODELS: List[str] = [
    "IGH_S5F_576",
    "IGH_S5F_576_Extended",
    "IGL_S5F_576",
    "TCRB_UNIFORM_576",
]

DEFAULT_CACHE_DIR: Path = Path.home() / ".alignair" / "models"


def _require_huggingface_hub():
    """Lazy import guard — gives a clear error if huggingface_hub is missing."""
    try:
        import huggingface_hub  # noqa: F401
        return huggingface_hub
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models. "
            "Install it with:  pip install 'alignair[cli]'  "
            "or:  pip install huggingface-hub"
        )


def list_available_models() -> List[str]:
    """Return the names of all pretrained model bundles available on the Hub."""
    return list(AVAILABLE_MODELS)


def is_model_cached(
    model_name: str,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> bool:
    """Check whether a model bundle is already downloaded locally."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    bundle_dir = cache_dir / model_name
    return bundle_dir.is_dir() and (bundle_dir / "config.json").exists()


def download_model(
    model_name: str,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download a pretrained model bundle from HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        One of the names from ``AVAILABLE_MODELS``.
    revision : str, optional
        Git tag / branch / commit on the HF repo (e.g. ``"v2.0.2"``).
        Defaults to the repo's main branch.
    cache_dir : Path, optional
        Local directory to store downloaded bundles.
        Defaults to ``~/.alignair/models``.
    force : bool
        Re-download even if the bundle already exists locally.

    Returns
    -------
    Path
        Absolute path to the downloaded bundle directory, ready for
        ``--model-dir`` or ``from_pretrained()``.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {', '.join(AVAILABLE_MODELS)}"
        )

    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    bundle_dir = cache_dir / model_name

    if not force and is_model_cached(model_name, revision, cache_dir):
        logger.info("Model '%s' already cached at %s", model_name, bundle_dir)
        return bundle_dir

    hf_hub = _require_huggingface_hub()

    logger.info("Downloading '%s' from %s ...", model_name, HF_REPO_ID)
    snapshot_path = hf_hub.snapshot_download(
        repo_id=HF_REPO_ID,
        revision=revision,
        allow_patterns=f"{model_name}/**",
        local_dir=cache_dir,
    )

    bundle_dir = Path(snapshot_path) / model_name
    if not (bundle_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Downloaded snapshot does not contain a valid bundle at "
            f"{bundle_dir}. Check that '{model_name}' exists in the "
            f"HuggingFace repo '{HF_REPO_ID}'."
        )

    logger.info("Model '%s' ready at %s", model_name, bundle_dir)
    return bundle_dir


def get_model_path(
    model_name: str,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Return the local path to a model bundle, downloading it if necessary.

    This is the main entry point for resolving a model name to a usable
    local directory.
    """
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)

    if is_model_cached(model_name, revision, cache_dir):
        return cache_dir / model_name

    return download_model(model_name, revision=revision, cache_dir=cache_dir)
