"""Pretrained-model hub: a small catalog of published AlignAIR bundles plus helpers to resolve
a model spec — a local path, a catalog id, or an ``org/name`` Hugging Face repo id — downloading
from the Hugging Face Hub when needed. A bundle is a directory of files (model.pt, config.json,
reference.json, …), which the Hub hosts and ``snapshot_download`` fetches as a local directory.
"""
from __future__ import annotations

import os

# Catalog of published bundles. `repo` is a Hugging Face Hub repo id that hosts the bundle files.
# This is the starting catalog; publishing a bundle to its repo makes `alignair model download
# <id>` / `alignair predict --model <id>` work for everyone.
MODEL_CATALOG = {
    "human-igh-ogrdb": {
        "repo": "AlignAIR/human-igh-ogrdb",
        "revision": None,
        "species": "human", "locus": "IGH",
        "description": "Human IGH (OGRDB reference) — general-purpose heavy-chain aligner.",
    },
}


def list_models() -> dict:
    return MODEL_CATALOG


def _download(repo: str, revision=None, dest=None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("error: downloading models needs huggingface_hub "
                         "— install with `pip install \"AlignAIR[cli]\"`")
    try:
        return snapshot_download(repo_id=repo, revision=revision, local_dir=dest)
    except Exception as e:                       # network / not-found / auth
        raise SystemExit(f"error: could not download '{repo}' from the Hugging Face Hub: {e}")


def resolve_model(spec: str, dest=None) -> str:
    """Return a local path (bundle dir or .pt checkpoint) for `spec`, downloading if needed.

    `spec` may be: a local path; a catalog id (see MODEL_CATALOG / `alignair model list`);
    or an ``org/name`` Hugging Face repo id (auto-downloaded)."""
    if os.path.exists(spec):
        return spec
    if spec in MODEL_CATALOG:
        e = MODEL_CATALOG[spec]
        return _download(e["repo"], e.get("revision"), dest)
    if "/" in spec:                              # looks like a Hugging Face repo id
        return _download(spec, None, dest)
    raise SystemExit(
        f"error: unknown model '{spec}'. Provide a local path, a catalog id "
        f"(see `alignair model list`), or an org/name Hugging Face repo id.")


def load_pretrained(spec: str, device: str = "cpu"):
    """Resolve + load a pretrained model. Returns the dict from load_dnalignair_bundle
    (config, reference_set/dataconfigs, locus, calibration, meta, model)."""
    from .serialization.dnalignair_bundle import load_dnalignair_bundle, is_bundle
    path = resolve_model(spec)
    if not is_bundle(path):
        raise SystemExit(f"error: '{spec}' resolved to {path}, which is not an AlignAIR bundle")
    return load_dnalignair_bundle(path, build=True, device=device)
