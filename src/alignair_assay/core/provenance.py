"""Provenance: the git state of a checkout, file and source-tree digests, the code identity of a Python package and
the GenAIRR build that simulated the reads."""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import subprocess
from collections.abc import Iterable
from pathlib import Path

# The files of a Python package that are its code: Python and Cython sources.
CODE_SUFFIXES = (".py", ".pyx")
# What ``code_identity`` records, in order.
CODE_IDENTITY = ("code_sha256", "torch_version")


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


def files_digest(root, paths: Iterable) -> str:
    """Hex SHA-256 identifying the content of the files ``paths`` under directory ``root``: over the files in the
    sorted order of their POSIX paths relative to ``root``, each path, its byte length and its bytes. An edited,
    renamed, added or removed file changes it; the order ``paths`` come in does not."""
    root = Path(root)
    digest = hashlib.sha256()
    for relative in sorted(Path(path).relative_to(root).as_posix() for path in paths):
        data = (root / relative).read_bytes()
        digest.update(f"{relative}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def tree_digest(root, suffixes: tuple[str, ...] | None = None) -> str:
    """``files_digest`` of every file under directory ``root`` (recursively) whose name ends with one of
    ``suffixes`` (every file when None), leaving out ``__pycache__`` directories and ``.pyc`` files."""
    root = Path(root)
    files = [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.relative_to(root).parts
             and path.suffix != ".pyc" and (suffixes is None or path.name.endswith(suffixes))]
    return files_digest(root, files)


def code_identity(package_dir) -> dict:
    """The code of the Python package in ``package_dir``, by content, and the torch it runs on: ``code_sha256``
    (``tree_digest`` of its ``CODE_SUFFIXES`` files, their relative paths and bytes, so any source edit changes it
    while byte code and other files do not) and the ``torch_version`` installed for this interpreter (None without
    torch). Read from the files each time it is called."""
    try:
        torch = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch = None
    return {"code_sha256": tree_digest(package_dir, suffixes=CODE_SUFFIXES), "torch_version": torch}


@functools.lru_cache(maxsize=None)
def _package_digest(directory: str) -> str:
    """``tree_digest`` of an imported package's directory, computed once per process: the process runs the code it
    imported, whatever happens to the files later."""
    return tree_digest(directory)


def genairr_provenance() -> dict:
    """The GenAIRR in use: ``genairr_version`` (installed metadata version), ``genairr_source`` (directory of the
    imported package), ``genairr_git_commit``/``genairr_git_dirty`` (the git state of the checkout holding that
    directory) and ``genairr_source_sha256`` (``tree_digest`` of that directory: its Python sources, compiled engine
    and data, computed once per process).

    The metadata version of an editable install can lag the source, so the commit identifies the simulator, and the
    source digest tells apart two states of a dirty checkout (an engine rebuilt in place). The git fields are None
    unless the package's own file is tracked by the checkout holding it: an install inside another project's
    checkout (a virtual environment under its tree, ignored or not) is not that project's code. Every field but the
    version is None when GenAIRR cannot be imported.
    """
    try:
        version = importlib.metadata.version("GenAIRR")
    except importlib.metadata.PackageNotFoundError:
        version = None
    git = {"git_commit": None, "git_dirty": None}
    try:
        import GenAIRR
    except ImportError:
        source = None
    else:
        package_file = Path(GenAIRR.__file__).resolve()
        source = package_file.parent
        git = package_git_state(package_file)
    return {"genairr_version": version, "genairr_source": None if source is None else str(source),
            "genairr_git_commit": git["git_commit"], "genairr_git_dirty": git["git_dirty"],
            "genairr_source_sha256": None if source is None else _package_digest(str(source))}


def package_git_state(package_file) -> dict:
    """``git_state`` of the checkout holding a package, when it tracks the package's own file ``package_file`` (its
    ``__init__.py``); both None otherwise: an install inside another project's checkout, ignored or not, is not
    that project's code."""
    package_file = Path(package_file)
    if _git(package_file.parent, "ls-files", "--error-unmatch", "--", package_file.name) is None:
        return {"git_commit": None, "git_dirty": None}
    return git_state(package_file.parent)


def _git(directory, *args: str) -> str | None:
    """Stripped stdout of a git command run in ``directory``, or None if it fails."""
    try:
        r = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None
