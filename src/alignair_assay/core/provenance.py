"""Provenance: the git state of a checkout, file digests and the GenAIRR build that simulated the reads."""
from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path


def git_state(directory) -> dict:
    """``git_commit`` (HEAD) and ``git_dirty`` (uncommitted changes) of the checkout holding ``directory``; both
    None outside a git checkout."""
    commit = _git(directory, "rev-parse", "HEAD")
    if not commit:
        return {"git_commit": None, "git_dirty": None}
    status = _git(directory, "status", "--porcelain")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def sha256_file(path) -> str:
    """Hex SHA-256 of the file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def genairr_provenance() -> dict:
    """The GenAIRR in use: ``genairr_version`` (installed metadata version), ``genairr_source`` (directory of the
    imported package) and ``genairr_git_commit``/``genairr_git_dirty`` (the git state of the checkout holding that
    directory).

    The metadata version of an editable install can lag the source, so the commit identifies the simulator. The git
    fields are None outside a checkout, and every field but the version is None when GenAIRR cannot be imported.
    """
    try:
        version = importlib.metadata.version("GenAIRR")
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        import GenAIRR
    except ImportError:
        source = None
        git = {"git_commit": None, "git_dirty": None}
    else:
        source = Path(GenAIRR.__file__).resolve().parent
        git = git_state(source)
    return {"genairr_version": version, "genairr_source": None if source is None else str(source),
            "genairr_git_commit": git["git_commit"], "genairr_git_dirty": git["git_dirty"]}


def _git(directory, *args: str) -> str | None:
    """Stripped stdout of a git command run in ``directory``, or None if it fails."""
    try:
        r = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None
