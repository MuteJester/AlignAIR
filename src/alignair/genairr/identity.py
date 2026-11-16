"""Which GenAIRR this process runs: the version it reports, its determinism version and the content digest of its
package (sources, compiled engine and data: ``tree_sha256`` of the package directory)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path


@functools.lru_cache(maxsize=None)
def tree_sha256(directory: str) -> str:
    """SHA-256 over every file under ``directory`` but ``__pycache__`` and ``.pyc``: in the sorted order of their
    POSIX paths relative to it, each path, its byte length and its bytes. Computed once per process."""
    root = Path(directory)
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
                   if path.is_file() and '__pycache__' not in path.relative_to(root).parts and path.suffix != '.pyc')
    digest = hashlib.sha256()
    for name in names:
        data = (root / name).read_bytes()
        digest.update(f'{name}\0{len(data)}\0'.encode())
        digest.update(data)
    return digest.hexdigest()


def genairr_source_sha256() -> str:
    """The content digest of the GenAIRR package this process imports (its sources, compiled engine and data), as
    ``train_args["genairr_source_sha256"]`` records it (``tree_sha256`` of the package directory). About 0.5 s, once
    per process."""
    import GenAIRR
    return tree_sha256(str(Path(GenAIRR.__file__).resolve().parent))


def recorded_genairr_source() -> str | None:
    """``genairr_source_sha256()``, or None, said on one line, when a file of the package cannot be read: the digest
    is provenance, and never fails a run."""
    try:
        return genairr_source_sha256()
    except OSError as error:
        print(f"GenAIRR source digest not recorded: {error}", flush=True)
        return None


@functools.lru_cache(maxsize=1)
def _identity() -> tuple:
    import GenAIRR
    try:                         # silent here: a training run reports an unreadable package once, in its train_args
        source = genairr_source_sha256()
    except OSError:
        source = None
    return getattr(GenAIRR, "__version__", None), getattr(GenAIRR, "DETERMINISM_VERSION", None), source


def genairr_identity() -> dict:
    """``genairr_version`` (as installed), ``genairr_determinism_version`` (``GenAIRR.DETERMINISM_VERSION``) and
    ``genairr_source_sha256`` (None when a file of the package cannot be read), computed once per process."""
    version, determinism, source = _identity()
    return {"genairr_version": version, "genairr_determinism_version": determinism, "genairr_source_sha256": source}
