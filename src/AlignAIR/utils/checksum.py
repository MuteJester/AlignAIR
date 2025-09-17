"""Checksum utilities."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_files(paths: Iterable[Path]) -> str:
    sha = hashlib.sha256()
    for p in paths:
        with p.open('rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha.update(chunk)
    return sha.hexdigest()

__all__ = ["sha256_files"]
