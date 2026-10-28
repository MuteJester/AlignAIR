"""``stored_runs``: where the gated tests find the assay's stored runs (read only)."""
from pathlib import Path

import pytest

from tests.alignair_assay.stored_runs import ENV, runs_root

pytestmark = pytest.mark.component


def _checkout(root: Path) -> Path:
    (root / ".private" / "assay").mkdir(parents=True)
    return root


def test_the_runs_are_the_checkouts_own(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    repo = _checkout(tmp_path / "repo")
    assert runs_root(repo) == repo / ".private" / "assay"


def test_a_worktree_without_runs_reads_the_main_checkouts(tmp_path, monkeypatch):
    # run data lives with the main checkout; a worktree's .git file names the main checkout's git directory
    monkeypatch.delenv(ENV, raising=False)
    main = _checkout(tmp_path / "main")
    worktree = tmp_path / "main" / "worktrees" / "w"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'w'}\n")
    assert runs_root(worktree) == main / ".private" / "assay"
    # without runs anywhere, the worktree's own (missing) directory: the gated tests skip
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / ".git").write_text(f"gitdir: {tmp_path / 'nowhere' / '.git' / 'worktrees' / 'b'}\n")
    assert runs_root(bare) == bare / ".private" / "assay"


def test_the_environment_names_the_runs(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / "elsewhere"))
    assert runs_root(_checkout(tmp_path / "repo")) == tmp_path / "elsewhere"
