"""A frozen copy of GenAIRR: the package as it is now, copied and named by its content, to run against instead of
the live checkout.

GenAIRR is installed editable from its checkout, where it is edited and committed while the assay runs: a build
records the GenAIRR it used, so a commit midway makes the run's builds stale, the tests flaky and dev numbers drift
from day to day. ``freeze_genairr(dest_root)`` copies the importable package (``package_source``: its sources,
compiled engine and data; byte code left out) into ``dest_root/<digest>/GenAIRR``, ``<digest>`` being the copy's
``core.provenance.tree_digest``, and returns ``dest_root/<digest>``: the root to put first on the path
(``PYTHONPATH=<root>:src``). An unchanged source is the same digest, so its copy is reused; a changed one gets a
directory of its own, and the old one stays as it was. The copy is made in a temporary directory under ``dest_root``
and renamed into place once it, and the source after it, digest as the source did before: a source that moved
while it was copied is copied again, at most ``ATTEMPTS`` times. Beside the package, ``FREEZE_RECORD`` holds what
was frozen: the source directory, its checkout's commit and dirty flag (the copy has no git state of its own), the
installed version and the digest.

Re-freeze deliberately (when a GenAIRR change should be picked up), never implicitly: tests, the audit, model
iteration and claims all run on one frozen copy. ``LIVE_ENV=1`` makes the tests run on the live GenAIRR instead.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.provenance import package_git_state, tree_digest

PACKAGE = "GenAIRR"
# Where frozen copies live, relative to the repository root (git-ignored).
DEFAULT_DEST = Path(".private") / "genairr_frozen"
FREEZE_RECORD = "freeze.json"
FREEZE_KIND = "alignair_assay.genairr_freeze"
FREEZE_VERSION = "1"
# How many copies are tried while the source keeps moving.
ATTEMPTS = 3
# Set to 1 to run the tests on the live GenAIRR instead of a frozen copy.
LIVE_ENV = "ALIGNAIR_ASSAY_GENAIRR_LIVE"


def package_source() -> Path:
    """The directory of the importable GenAIRR package (``importlib.util.find_spec``; a frozen copy first on the
    path is the one found). ModuleNotFoundError when GenAIRR cannot be found."""
    spec = importlib.util.find_spec(PACKAGE)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"{PACKAGE} is not importable: nothing to freeze")
    return Path(list(spec.submodule_search_locations)[0]).resolve()


def freeze_genairr(dest_root, *, source=None, attempts: int = ATTEMPTS) -> Path:
    """Freeze GenAIRR (``source``, default ``package_source()``) into ``dest_root`` as the module docstring says;
    returns the root to put first on ``PYTHONPATH`` (``dest_root/<digest>``, holding ``GenAIRR/`` and
    ``FREEZE_RECORD``). An existing copy of the same digest is reused once it is checked to still hold that content
    (RuntimeError when it was edited in place: remove it and freeze again). RuntimeError when the source changed
    while it was copied on every one of ``attempts`` tries; nothing is left behind but complete copies."""
    dest = Path(dest_root).expanduser().resolve()
    source = Path(source if source is not None else package_source()).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        digest = tree_digest(source)
        root = dest / digest
        if root.exists():
            return _checked(root, digest)
        git = package_git_state(source / "__init__.py")
        work = Path(tempfile.mkdtemp(prefix=".freeze-", dir=dest))
        try:
            _copy_tree(source, work / PACKAGE)
            if tree_digest(work / PACKAGE) != digest or tree_digest(source) != digest:
                continue  # the source moved while it was copied: copy it again
            record = {"kind": FREEZE_KIND, "version": FREEZE_VERSION, "source": str(source),
                      "genairr_version": _installed_version(), "genairr_git_commit": git["git_commit"],
                      "genairr_git_dirty": git["git_dirty"], "genairr_source_sha256": digest,
                      "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (work / FREEZE_RECORD).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            work.chmod(0o755)  # mkdtemp makes it private to its creator
            try:
                os.rename(work, root)
            except OSError:
                if not root.exists():
                    raise
                return _checked(root, digest)  # frozen by another process meanwhile
            return root
        finally:
            shutil.rmtree(work, ignore_errors=True)
    raise RuntimeError(f"{PACKAGE}'s source {source} changed while it was copied, {attempts} times: freeze it once "
                       "it is no longer being edited")


def read_freeze(root) -> dict | None:
    """The ``FREEZE_RECORD`` of the frozen root ``root``; None without a readable one."""
    try:
        return json.loads((Path(root) / FREEZE_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def frozen_record(package_dir) -> dict | None:
    """What was frozen, when the GenAIRR package directory ``package_dir`` is a frozen copy (its root's
    ``FREEZE_RECORD``, whose digest names the root); None for a live package."""
    if package_dir is None:
        return None
    root = Path(package_dir).resolve().parent
    record = read_freeze(root)
    return record if record is not None and record.get("genairr_source_sha256") == root.name else None


def _checked(root: Path, digest: str) -> Path:
    """``root`` when its copy still holds the content ``digest`` names; RuntimeError otherwise."""
    if not (root / PACKAGE).is_dir() or tree_digest(root / PACKAGE) != digest:
        raise RuntimeError(f"the frozen copy {root} no longer holds the content its name digests (it was edited or "
                           "removed in place): remove that directory and freeze again")
    return root


def _copy_tree(source: Path, target: Path) -> None:
    """Copy the package ``source`` to ``target``, without byte code (``tree_digest`` leaves it out)."""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return None
