"""Publish individual files through a temporary sibling and atomic replacement."""
import os
import json
from pathlib import Path
import tempfile

from ..errors import InputError, OperationIOError, error_context


def atomic_write(path, writer, *, stage="artifact_write"):
    """Run writer(temporary_path); failure preserves the previous file and removes the temporary.

    File contents are fsynced before replacement. This is a per-file operation, not a directory
    transaction or a guarantee of directory-entry persistence after power loss. Writer/OS exceptions
    retain their original types; domain errors inherit the supplied stage and destination context.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        with error_context(stage=stage, path=str(path)):
            writer(temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path, text, *, stage="artifact_write"):
    """Publish UTF-8 text using the same file lifecycle as model checkpoints."""
    def write(temporary):
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(text)
    atomic_write(path, write, stage=stage)


def atomic_write_json(path, value, *, sort_keys=True, stage="artifact_write"):
    """Publish finite JSON, with contextual encoding and I/O errors and no partial replacement.

    Encoding completes before any output is created. Formatting uses two-space indentation and
    the standard JSON Unicode escaping policy. This shares atomic_write's per-file guarantees.
    """
    with error_context(stage=stage, path=str(path)):
        try:
            text = json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise InputError("artifact requires finite JSON values") from exc
        try:
            atomic_write_text(path, text, stage=stage)
        except OSError as exc:
            raise OperationIOError("cannot publish JSON artifact") from exc
