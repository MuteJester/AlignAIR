import importlib.metadata
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from alignair_assay.cli import main
from alignair_assay.core.jsonl import read_header, write_jsonl
from alignair_assay.core.provenance import (
    CODE_IDENTITY,
    code_identity,
    files_digest,
    genairr_provenance,
    git_state,
    sha256_file,
    tree_digest,
)

pytestmark = pytest.mark.component

GENAIRR_FIELDS = {"genairr_version", "genairr_source", "genairr_git_commit", "genairr_git_dirty",
                  "genairr_source_sha256"}


def test_genairr_provenance_fields(genairr):
    prov = genairr_provenance()
    assert set(prov) == GENAIRR_FIELDS
    assert prov["genairr_version"] == importlib.metadata.version("GenAIRR")
    assert prov["genairr_source"].endswith("GenAIRR")
    assert prov["genairr_source"] == str(Path(genairr.__file__).resolve().parent)
    commit = prov["genairr_git_commit"]
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)
    assert prov["genairr_git_dirty"] in ((None,) if commit is None else (True, False))
    # The package's content, compiled engine included: a dirty checkout rebuilt in place is another simulator.
    assert prov["genairr_source_sha256"] == tree_digest(prov["genairr_source"])


def test_files_digest_tracks_names_and_bytes(tmp_path):
    (tmp_path / "pkg").mkdir()
    a, b = tmp_path / "pkg" / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    digest = files_digest(tmp_path, [b, a])
    assert digest == files_digest(tmp_path, [a, b])  # order-free
    a.write_text("x = 2\n")
    assert files_digest(tmp_path, [a, b]) != digest  # an edit
    a.write_text("x = 1\n")
    assert files_digest(tmp_path, [a, b]) == digest
    renamed = tmp_path / "pkg" / "c.py"
    a.rename(renamed)
    assert files_digest(tmp_path, [renamed, b]) != digest  # a rename
    assert files_digest(tmp_path, [b]) != files_digest(tmp_path, [renamed, b])  # a removal


def test_tree_digest_skips_bytecode(tmp_path):
    (tmp_path / "sub" / "__pycache__").mkdir(parents=True)
    (tmp_path / "sub" / "m.py").write_text("x = 1\n")
    (tmp_path / "ext.pyx").write_text("cdef int x\n")
    (tmp_path / "notes.txt").write_text("n\n")
    digest = tree_digest(tmp_path, suffixes=(".py", ".pyx"))
    assert digest == files_digest(tmp_path, [tmp_path / "sub" / "m.py", tmp_path / "ext.pyx"])
    (tmp_path / "sub" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "notes.txt").write_text("changed\n")
    assert tree_digest(tmp_path, suffixes=(".py", ".pyx")) == digest
    assert tree_digest(tmp_path) == files_digest(tmp_path, [tmp_path / "sub" / "m.py", tmp_path / "ext.pyx",
                                                            tmp_path / "notes.txt"])


@pytest.mark.parametrize("ignored", [True, False])
def test_genairr_git_state_only_from_a_checkout_that_tracks_it(tmp_path, monkeypatch, ignored):
    # A non-editable install inside another project's checkout (a venv under the AlignAIR repo, say) is not that
    # checkout's code: its HEAD and dirty flag must not be reported as GenAIRR's.
    repo = tmp_path / "repo"
    package = repo / ".venv" / "site-packages" / "GenAIRR"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (repo / ".gitignore").write_text(".venv/\n" if ignored else "")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"]):
        subprocess.run(git + args, check=True, capture_output=True)
    head = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    fake = types.ModuleType("GenAIRR")
    fake.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "GenAIRR", fake)

    prov = genairr_provenance()
    assert prov["genairr_source"] == str(package.resolve())
    assert prov["genairr_source_sha256"] == tree_digest(package)
    if ignored:
        assert (prov["genairr_git_commit"], prov["genairr_git_dirty"]) == (None, None)
    else:
        assert (prov["genairr_git_commit"], prov["genairr_git_dirty"]) == (head, False)


def test_truth_header_records_genairr_commit(tmp_path, load_fixture, reference):
    cases = tmp_path / "cases.jsonl"
    write_jsonl([load_fixture("amplicon_janchor_000000.json")], cases)
    out = tmp_path / "truth.jsonl.gz"
    assert main(["truth", "--bench-cases", str(cases), "--out", str(out)]) == 0
    header = read_header(out)
    assert header["reference_digest"] == reference.digest()
    assert {k: header[k] for k in GENAIRR_FIELDS} == genairr_provenance()


def test_sha256_file(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_git_state_reports_dirty_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q"], check=True)
    (repo / "f.txt").write_text("a\n")
    subprocess.run(git + ["add", "f.txt"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "init"], check=True)
    state = git_state(repo)
    assert len(state["git_commit"]) == 40 and state["git_dirty"] is False
    (repo / "f.txt").write_text("b\n")
    assert git_state(repo)["git_dirty"] is True
    assert git_state(tmp_path / "missing") == {"git_commit": None, "git_dirty": None}


def test_code_identity_is_the_python_and_cython_sources(tmp_path):
    # One implementation for the alignair-local runner and the in-process evaluator.
    package = tmp_path / "pkg"
    (package / "sub").mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    (package / "sub" / "b.pyx").write_text("cdef int y\n")
    identity = code_identity(package)
    assert tuple(identity) == CODE_IDENTITY == ("code_sha256", "torch_version")
    assert identity == {"code_sha256": tree_digest(package, suffixes=(".py", ".pyx")),
                        "torch_version": importlib.metadata.version("torch")}
    # Notes and byte code are not code; an edited source is.
    (package / "notes.md").write_text("n\n")
    (package / "sub" / "__pycache__").mkdir()
    (package / "sub" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"\0")
    assert code_identity(package) == identity
    (package / "a.py").write_text("x = 2\n")
    assert code_identity(package)["code_sha256"] != identity["code_sha256"]
