"""Loading a model straight from its own Hugging Face Hub repo.

Besides the catalog path (``registry.json`` under a source), ``Aligner.from_pretrained("hf://org/repo")`` or
``"org/repo"`` pulls the ``.alignair`` from a Hub model repo through ``huggingface_hub``, with revision (branch, tag or
commit) pinning, tokens for private or gated repos, offline cache lookup, the standard HF cache, retries and resumable
transfers. Both paths go through ``resolve_model``.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..errors import MissingDependencyError, RegistryError

DEFAULT_FILENAME = "model.alignair"


def is_hf_repo_spec(spec: str) -> bool:
    """True if ``spec`` names a HF model repo (``hf://org/repo[/file][@rev]`` or a bare ``org/repo[@rev]``) rather
    than a local path or a catalog id. A bare spec with a file below the repo, or one naming a ``.alignair`` file, is
    a path: only the ``hf://`` form reaches a file inside a repo."""
    if not spec or os.path.exists(spec):
        return False
    if spec.startswith("hf://"):
        return True
    if "://" in spec or names_model_file(spec):          # some other URL scheme (http/file), or a model file's path
        return False
    body = spec.partition("@")[0]
    return body.count("/") == 1 and not spec.startswith((".", "/", "~"))


def names_model_file(spec: str) -> bool:
    """True if a bare ``spec`` (no ``hf://``) names a model file: it ends in ``.alignair``, or has a path below
    ``org/repo``."""
    body = spec.partition("@")[0]
    return body.endswith(".alignair") or body.count("/") > 1


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


def hf_token(explicit: str | None) -> str | None:
    """The Hugging Face token: ``explicit``, else ``$HF_TOKEN``, else ``$HUGGING_FACE_HUB_TOKEN``."""
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
    the local cache directly (no network; a cache miss raises a clear error). Requires ``huggingface_hub``."""
    repo_id, spec_rev, spec_file = parse_hf_spec(spec)
    rev = revision or spec_rev
    fname = filename or spec_file or DEFAULT_FILENAME
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError as e:                             # pragma: no cover - optional dep
        raise MissingDependencyError(
            "loading a model directly from a Hugging Face repo needs 'huggingface_hub'; "
            "install it with `pip install alignair[hub]` (or `[cli]`).",
            context={"dependency": "huggingface_hub", "stage": "model_download"}) from e
    from .. import __version__
    try:
        if offline:
            # Some SDK versions build telemetry headers even with local_files_only.
            # The cache lookup API does not create a request or consult remote metadata.
            cached = try_to_load_from_cache(repo_id=repo_id, filename=fname, revision=rev)
            if isinstance(cached, str) and Path(cached).is_file():
                return Path(cached)
            raise FileNotFoundError(f"{repo_id}@{rev or 'main'}:{fname} is not in the local HF cache")
        path = hf_hub_download(repo_id=repo_id, filename=fname, revision=rev, token=hf_token(token),
                               library_name="alignair", library_version=__version__)
    except Exception as e:                               # normalize hub errors to an actionable message
        hint = " (offline: not in the local HF cache)" if offline else ""
        raise RegistryError(
            f"could not fetch {fname!r} from Hugging Face repo {repo_id!r} "
            f"(revision {rev or 'main'}){hint}",
            context={"stage": "model_download", "repo_id": repo_id, "revision": rev,
                     "filename": fname, "offline": offline},
            hint="Check repository access, revision, connectivity, and local cache availability.") from e
    return Path(path)
