"""A model file's ``git_commit`` is the commit of the AlignAIR checkout that wrote it, never that of the repository
the user happens to run from; an installed AlignAIR (no checkout of its own) records none."""
import subprocess
from pathlib import Path

import pytest

import alignair
from alignair.model_file import _provenance

pytestmark = pytest.mark.component

PACKAGE = Path(alignair.__file__).resolve().parent


def _git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()


def test_the_commit_is_alignairs_own_not_the_working_directorys(tmp_path, monkeypatch):
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / "f").write_text("x")
    _git("add", "f", cwd=tmp_path)
    _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "user's own work", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    commit = _provenance(publish=True)["git_commit"]
    assert commit != _git("rev-parse", "HEAD", cwd=tmp_path)
    assert commit == _git("rev-parse", "HEAD", cwd=PACKAGE)


def test_an_alignair_outside_any_checkout_records_no_commit(tmp_path, monkeypatch):
    _git("init", "-q", cwd=tmp_path)          # a repository that holds the package's directory but not its files
    monkeypatch.setattr("alignair.model_file._package_dir", lambda: tmp_path)
    assert _provenance(publish=True)["git_commit"] is None
