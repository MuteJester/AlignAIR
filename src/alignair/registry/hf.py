"""Direct Hugging Face Hub model loading.

The catalog path (``registry.json`` under a source) still works; this adds the *one-repo-per-model*
path used by `Aligner.from_pretrained("hf://org/repo")` / `"org/repo"`: pull the ``.alignair`` straight
from a HF model repo via ``huggingface_hub`` — with revision (branch/tag/commit) pinning, tokens for
private/gated repos, offline (``local_files_only``) mode, the standard HF cache, retries and resumable
transfers. Both paths funnel through ``resolve_model`` so aliases stay a convenience, not a second
incompatible loader.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_FILENAME = "model.alignair"


def is_hf_repo_spec(spec: str) -> bool:
    """True if ``spec`` names a HF model repo (``hf://org/repo`` or a bare ``org/repo``) rather than a
    local path or a catalog id. A local path or a plain id (no ``/``) is not a HF repo spec."""
    if not spec or os.path.exists(spec):
        return False
    if spec.startswith("hf://"):
        return True
    if "://" in spec:                                    # some other URL scheme (http/file)
        return False
    return "/" in spec and not spec.startswith((".", "/", "~"))


def parse_hf_spec(spec: str) -> tuple[str, str | None, str | None]:
    """Parse ``hf://org/repo[/sub/file.alignair][@rev]`` (or the ``hf://``-less form) into
    ``(repo_id, revision, filename)``. ``repo_id`` is the first two path segments; anything after is a
    file path within the repo (default ``model.alignair``); ``@rev`` pins a branch/tag/commit."""
    s = spec[len("hf://"):] if spec.startswith("hf://") else spec
    body, _, rev = s.partition("@")
    parts = [p for p in body.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"not a Hugging Face repo spec: {spec!r} (expected org/repo)")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) or None
    return repo_id, (rev or None), filename


def _token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolved_commit(path: str | Path) -> str | None:
    """Extract the resolved commit SHA from a downloaded HF cache path
    (``.../snapshots/<sha>/<file>``) for run provenance; None if not a snapshot path."""
    parts = Path(path).parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def download_from_hub(spec: str, *, revision: str | None = None, filename: str | None = None,
                      token: str | None = None, offline: bool = False) -> Path:
    """Download a model's ``.alignair`` from a HF repo and return the local cached path. ``offline`` uses
    ``local_files_only`` (no network; a cache miss raises a clear error). Requires ``huggingface_hub``."""
    repo_id, spec_rev, spec_file = parse_hf_spec(spec)
    rev = revision or spec_rev
    fname = filename or spec_file or DEFAULT_FILENAME
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:                             # pragma: no cover - optional dep
        raise ImportError(
            "loading a model directly from a Hugging Face repo needs 'huggingface_hub'; "
            "install it with `pip install alignair[hub]` (or `[cli]`).") from e
    try:
        from .. import __version__ as _v
    except Exception:                                    # noqa: BLE001
        _v = "0"
    try:
        path = hf_hub_download(repo_id=repo_id, filename=fname, revision=rev, token=_token(token),
                               local_files_only=offline, library_name="alignair", library_version=_v)
    except Exception as e:                               # normalize hub errors to an actionable message
        hint = " (offline: not in the local HF cache)" if offline else ""
        raise ValueError(
            f"could not fetch {fname!r} from Hugging Face repo {repo_id!r} "
            f"(revision {rev or 'main'}){hint}: {e}") from e
    return Path(path)
