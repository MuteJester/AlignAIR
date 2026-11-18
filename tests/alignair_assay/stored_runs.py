"""Where the assay's stored runs are: the run directories (``<runs>/<run>``) some gated tests read, never write.

Run data lives with the main checkout (``.private/assay``), so a git worktree without its own reads the main
checkout's; ``ALIGNAIR_ASSAY_RUNS`` names another directory.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Set to the directory holding the stored runs (default: see ``runs_root``).
ENV = "ALIGNAIR_ASSAY_RUNS"


def runs_root(repo: Path = REPO) -> Path:
    """``ALIGNAIR_ASSAY_RUNS`` when set; else ``repo/.private/assay`` when it exists; else, for a git worktree (whose
    ``.git`` file names ``<main>/.git/worktrees/<name>``), the main checkout's ``.private/assay`` when it exists; else
    ``repo/.private/assay`` (missing: the tests that need a stored run skip)."""
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    own = repo / ".private" / "assay"
    if own.is_dir():
        return own
    git = repo / ".git"
    if git.is_file():
        gitdir = git.read_text(encoding="utf-8").partition("gitdir:")[2].strip()
        if gitdir:
            main = Path(gitdir).parents[2] / ".private" / "assay"
            if main.is_dir():
                return main
    return own


RUNS = runs_root()
