"""Publish individual files through a temporary sibling and atomic replacement."""
import os
from pathlib import Path
import tempfile

from ..errors import error_context


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
