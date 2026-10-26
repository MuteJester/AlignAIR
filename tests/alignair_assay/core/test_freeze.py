"""``core.freeze``: a frozen, content-addressed copy of GenAIRR (``freeze_genairr``) and the command that makes it.

The copy is named by its content digest (``core.provenance.tree_digest``), so an unchanged source is reused and a
changed one gets a directory of its own; a source that moves while it is copied is copied again.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alignair_assay.cli import main
from alignair_assay.core import freeze
from alignair_assay.core.freeze import FREEZE_RECORD, PACKAGE, freeze_genairr, read_freeze
from alignair_assay.core.provenance import tree_digest


def _package(root: Path) -> Path:
    """A fake GenAIRR package: sources, a compiled engine, data and byte code."""
    package = root / PACKAGE
    (package / "sub").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    (package / "__init__.py").write_text("VERSION = 1\n")
    (package / "sub" / "steps.py").write_text("x = 1\n")
    (package / "_engine.abi3.so").write_bytes(b"\x7fELF engine")
    (package / "sub" / "data.json").write_text("{}\n")
    (package / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"\0")
    return package


def _leftovers(dest: Path) -> list[str]:
    return sorted(path.name for path in dest.iterdir() if path.name.startswith("."))


def test_a_frozen_copy_is_named_by_its_content_and_reused_while_unchanged(tmp_path, monkeypatch):
    source = _package(tmp_path / "src")
    dest = tmp_path / "frozen"
    root = freeze_genairr(dest, source=source)
    digest = tree_digest(source)
    assert root == dest.resolve() / digest
    copy = root / PACKAGE
    assert tree_digest(copy) == digest and (copy / "_engine.abi3.so").read_bytes() == b"\x7fELF engine"
    assert not (copy / "__pycache__").exists()  # byte code is not content
    record = read_freeze(root)
    assert record["genairr_source_sha256"] == digest and record["source"] == str(source.resolve())
    assert _leftovers(dest) == []
    # unchanged: the same directory, not copied again
    copies = []
    monkeypatch.setattr(freeze, "_copy_tree", lambda *args: copies.append(args))
    assert freeze_genairr(dest, source=source) == root and copies == []


def test_a_changed_source_gets_a_directory_of_its_own(tmp_path):
    source = _package(tmp_path / "src")
    dest = tmp_path / "frozen"
    first = freeze_genairr(dest, source=source)
    (source / "sub" / "steps.py").write_text("x = 2\n")
    second = freeze_genairr(dest, source=source)
    assert second != first and second.name == tree_digest(source)
    assert (first / PACKAGE / "sub" / "steps.py").read_text() == "x = 1\n"  # the old copy is kept as it was
    assert (second / PACKAGE / "sub" / "steps.py").read_text() == "x = 2\n"


def test_a_source_that_moves_during_the_copy_is_copied_again(tmp_path, monkeypatch):
    source = _package(tmp_path / "src")
    dest = tmp_path / "frozen"
    copy_tree = freeze._copy_tree
    calls = []

    def moving(src, target):  # another process commits to GenAIRR while the first copy runs
        copy_tree(src, target)
        calls.append(target)
        if len(calls) == 1:
            (source / "sub" / "steps.py").write_text("x = 3\n")

    monkeypatch.setattr(freeze, "_copy_tree", moving)
    root = freeze_genairr(dest, source=source)
    assert len(calls) == 2 and root.name == tree_digest(source)
    assert (root / PACKAGE / "sub" / "steps.py").read_text() == "x = 3\n"
    assert _leftovers(dest) == [] and [path.name for path in dest.iterdir()] == [root.name]


def test_a_source_that_keeps_moving_is_refused(tmp_path, monkeypatch):
    source = _package(tmp_path / "src")
    dest = tmp_path / "frozen"
    copy_tree = freeze._copy_tree
    calls = []

    def moving(src, target):
        copy_tree(src, target)
        calls.append(target)
        (source / "sub" / "steps.py").write_text(f"x = {10 + len(calls)}\n")

    monkeypatch.setattr(freeze, "_copy_tree", moving)
    with pytest.raises(RuntimeError, match="changed while it was copied, 3 times"):
        freeze_genairr(dest, source=source)
    assert len(calls) == freeze.ATTEMPTS and list(dest.iterdir()) == []


def test_a_frozen_copy_edited_in_place_is_refused(tmp_path):
    source = _package(tmp_path / "src")
    dest = tmp_path / "frozen"
    root = freeze_genairr(dest, source=source)
    (root / PACKAGE / "sub" / "steps.py").write_text("x = 99\n")
    with pytest.raises(RuntimeError, match="no longer holds"):
        freeze_genairr(dest, source=source)


def test_the_freeze_records_the_checkouts_commit_and_dirty_flag(tmp_path):
    repo = tmp_path / "repo"
    source = _package(repo / "src")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(git + args, check=True, capture_output=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    record = read_freeze(freeze_genairr(tmp_path / "frozen", source=source))
    assert (record["genairr_git_commit"], record["genairr_git_dirty"]) == (head, False)
    (source / "sub" / "steps.py").write_text("x = 5\n")
    record = read_freeze(freeze_genairr(tmp_path / "frozen", source=source))
    assert (record["genairr_git_commit"], record["genairr_git_dirty"]) == (head, True)


def test_the_freeze_command_prints_the_root_and_the_pythonpath(tmp_path, monkeypatch, capsys):
    source = _package(tmp_path / "src")
    monkeypatch.setattr(freeze, "package_source", lambda: source)
    dest = tmp_path / "frozen"
    assert main(["freeze-genairr", "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    root = dest.resolve() / tree_digest(source)
    assert f"frozen GenAIRR: {root}" in out
    assert "commit: none (not a git checkout), dirty: unknown" in out
    assert f"PYTHONPATH={root}:src" in out
    assert json.loads((root / FREEZE_RECORD).read_text())["genairr_source_sha256"] == root.name


def test_the_package_source_is_the_importable_genairr(genairr):
    assert freeze.package_source() == Path(genairr.__file__).resolve().parent


@pytest.mark.skipif(os.environ.get("ALIGNAIR_ASSAY_GENAIRR_LIVE") == "1", reason="the session runs live GenAIRR")
def test_the_test_session_and_its_subprocesses_run_the_frozen_copy(genairr):
    # tests/alignair_assay/conftest.py freezes GenAIRR and puts the copy first, for this process and its children
    frozen = Path(__file__).resolve().parents[3] / ".private" / "genairr_frozen"
    package = Path(genairr.__file__).resolve().parent
    assert package.parent.parent == frozen and package.parent.name == tree_digest(package)
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(package.parent)
    child = subprocess.run([sys.executable, "-c", "import GenAIRR; print(GenAIRR.__file__)"], check=True,
                           capture_output=True, text=True).stdout.strip()
    assert Path(child).resolve().parent == package
