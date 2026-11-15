"""File digests."""
from __future__ import annotations

import hashlib
import os

_CHUNK = 1 << 20


def sha256_file(path: str | os.PathLike[str]) -> str:
    """The SHA-256 hex digest of the file at ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
