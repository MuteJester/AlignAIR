"""Verified, atomic local model cache + the ``resolve_model`` entry point.

Downloads stream to a ``.part`` temp under an exclusive lock, are SHA256-verified, and only then
atomically renamed into place — a killed or corrupt download never yields a usable cached model.
``resolve_model`` turns a filesystem path OR a ``id`` / ``id@version`` spec into a local artifact
path, hitting the network only when necessary (never for a pinned, already-cached version).
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

from . import sources


class IntegrityError(RuntimeError):
    """A downloaded artifact failed SHA256 verification."""


def cache_root() -> Path:
    if os.environ.get("ALIGNAIR_CACHE_DIR"):
        return Path(os.environ["ALIGNAIR_CACHE_DIR"])
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(xdg) / "alignair"


def cache_path(model_id: str, version: str) -> Path:
    return cache_root() / "models" / model_id / f"{version}.alignair"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:                                            # pragma: no cover - non-posix
        yield
        return
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _open_stream(source: str, relpath: str, offline: bool, revision: str = "main"):
    url = sources.artifact_url(source, relpath, revision=revision)
    local = sources._local_path(url)
    if local is not None:
        return open(local, "rb")
    if offline:
        raise sources.OfflineError(f"offline: cannot fetch {url}")
    import urllib.request
    return urllib.request.urlopen(url, timeout=60)                 # nosec - user-configured registry


def download_verified(source: str, relpath: str, dest: Path, expected_sha256: str | None = None, *,
                      offline: bool = False) -> Path:
    """Stream ``relpath`` from ``source`` into ``dest``, SHA256-verify, atomically install. Idempotent:
    a present, matching ``dest`` is returned without re-downloading."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    with _lock(dest.with_name(dest.name + ".lock")):
        if dest.exists() and (expected_sha256 is None or _sha256_file(dest) == expected_sha256):
            return dest
        h = hashlib.sha256()
        with _open_stream(source, relpath, offline) as r, open(part, "wb") as w:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                w.write(chunk)
        got = h.hexdigest()
        if expected_sha256 and got != expected_sha256:
            part.unlink(missing_ok=True)
            raise IntegrityError(
                f"SHA256 mismatch for {relpath}: expected {expected_sha256}, got {got} — refusing to install.")
        os.replace(part, dest)                                     # atomic
    return dest


def resolve_model(spec: str, *, sources: list[str] | None = None, offline: bool = False) -> Path:
    """A filesystem path -> that path; a ``id`` / ``id@version`` -> the verified cached artifact
    (downloading if absent). A pinned + already-cached version returns immediately (no network)."""
    if os.path.exists(spec):
        return Path(spec)
    model_id, _, version = spec.partition("@")
    version = version or None
    srcs = sources if sources is not None else _resolve_sources()
    if version:                                                    # pinned + cached -> no network
        cached = cache_path(model_id, version)
        if cached.exists():
            return cached
    found = _find(model_id, version, srcs, offline)
    if not found:
        known = _known_ids(srcs, offline)
        hint = f" Known ids: {', '.join(known)}." if known else ""
        raise ValueError(f"unknown model '{spec}' — pass a file path or a registry id.{hint}")
    src, ver, entry, _ = found
    dest = cache_path(model_id, ver)
    sha = entry.get("artifact_sha256")
    if dest.exists() and (sha is None or _sha256_file(dest) == sha):
        return dest
    return download_verified(src, entry["file"], dest, sha, offline=offline)


def _resolve_sources() -> list[str]:
    return sources.resolve_sources()


def _find(model_id, version, srcs, offline):
    return sources.find_model(model_id, version, srcs, offline=offline)


def _known_ids(srcs, offline) -> list[str]:
    ids: list[str] = []
    for src in srcs:
        try:
            ids += list(sources.load_registry(src, offline=offline).get("models", {}))
        except Exception:
            continue
    return sorted(set(ids))
