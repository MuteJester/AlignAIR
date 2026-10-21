"""The record helpers every workflow step shares: the digest that keys its caches, and strict JSON reads and atomic
writes of its records.

- ``digest``: hex SHA-256 of the canonical JSON of a record (keys sorted, no spaces, ASCII escapes): what every
  fingerprint and config hash of the assay is. It must not change, or every cache keyed on it goes stale.
- ``read_json``: the JSON object in a file; None when the file is missing, unreadable or holds no object.
- ``write_text`` and ``write_json``: written through a temporary file of the writer's own in the same directory,
  then renamed over the target, so no reader ever sees half a file, two writers never share a temporary file, and a
  failed write leaves the old file as it was and no temporary file behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(fields) -> str:
    """Hex SHA-256 of the canonical JSON (sorted keys, no spaces) of ``fields``."""
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path) -> dict | None:
    """The JSON object in ``path``; None when it is missing, unreadable or not an object."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_json(path, value, *, allow_nan: bool = False) -> None:
    """Write ``value`` as JSON (indented by 2, a final newline; strict unless ``allow_nan``) atomically
    (``write_text``)."""
    write_text(path, json.dumps(value, indent=2, allow_nan=allow_nan) + "\n")


def write_text(path, text: str) -> None:
    """Write ``text`` through a temporary file of its own in ``path``'s directory (made when missing), then rename it
    over ``path`` (mode 0644). The temporary file is removed when the write fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.chmod(temporary, 0o644)  # a temporary file is private; the record is not
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
