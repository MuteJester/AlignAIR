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
    """The local model cache root. Honors ``ALIGNAIR_CACHE_DIR``; otherwise the correct per-OS user
    cache dir via ``platformdirs`` (``~/.cache/alignair`` on Linux honoring ``XDG_CACHE_HOME``,
    ``~/Library/Caches/alignair`` on macOS, ``%LOCALAPPDATA%\\alignair`` on Windows), falling back to
    the XDG/home convention if ``platformdirs`` is unavailable."""
    if os.environ.get("ALIGNAIR_CACHE_DIR"):
        return Path(os.environ["ALIGNAIR_CACHE_DIR"])
    try:
        import platformdirs
        return Path(platformdirs.user_cache_dir("alignair"))
    except ImportError:                                            # pragma: no cover - stdlib fallback
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
def _excl_lock(path: Path, *, timeout: float = 600.0, stale_after: float = 3600.0, poll: float = 0.05):
    """Portable cross-process mutual exclusion via an atomic ``O_EXCL`` lockfile (works where ``flock``
    is absent, e.g. Windows — the previous no-op there let concurrent downloads corrupt the cache).
    Acquire by exclusive create; spin-wait while another holder has it; reclaim a *stale* lock (a dead
    holder's file older than ``stale_after``) so a crash can't deadlock the cache; time out otherwise."""
    import time
    fd = None
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(path)
            except OSError:
                age = 0.0
            if age > stale_after:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out after {timeout}s acquiring cache lock {path}")
            time.sleep(poll)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


@contextlib.contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:                                            # non-posix (Windows): portable protocol
        with _excl_lock(path):
            yield
        return
    with open(path, "w") as f:                                     # POSIX: robust, auto-released on death
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


def resolve_model(spec: str, *, sources: list[str] | None = None, offline: bool = False,
                  token: str | None = None, revision: str | None = None) -> Path:
    """Resolve a model spec to a local artifact path:

    * a filesystem path -> that path;
    * a Hugging Face repo (``hf://org/repo`` or ``org/repo``) -> the ``.alignair`` pulled straight from
      that repo via ``huggingface_hub`` (revision/token/offline honored — the one-repo-per-model path);
    * a catalog ``id`` / ``id@version`` -> the verified cached artifact from a registry source.

    A pinned + already-cached version returns immediately (no network)."""
    if os.path.exists(spec):
        return Path(spec)
    from . import hf
    if hf.is_hf_repo_spec(spec):                          # direct one-repo-per-model HF loading (P0-11)
        return hf.download_from_hub(spec, revision=revision, token=token, offline=offline)
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


def installed_models() -> dict[str, list[str]]:
    """{model_id: [installed versions]} scanned from the cache dir."""
    root = cache_root() / "models"
    out: dict[str, list[str]] = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir():
            vers = sorted((p.name[: -len(".alignair")] for p in d.glob("*.alignair")), key=_version_key)
            if vers:
                out[d.name] = vers
    return out


def _version_key(v: str):
    try:
        return (0, tuple(int(x) for x in v.split(".")))
    except ValueError:
        return (1, v)


def prune(keep: int = 1) -> list[Path]:
    """Remove all but the newest ``keep`` cached versions per model. Returns the removed paths."""
    removed: list[Path] = []
    for mid, vers in installed_models().items():
        drop = vers if keep <= 0 else vers[:-keep]
        for ver in drop:
            p = cache_path(mid, ver)
            try:
                p.unlink()
                removed.append(p)
            except OSError:
                pass
    return removed


def verify_installed(srcs: list[str], *, offline: bool = False, model_id: str | None = None):
    """Re-hash each installed artifact against the registry. Yields (id, version, ok|None-if-unknown)."""
    out = []
    for mid, vers in installed_models().items():
        if model_id and mid != model_id:
            continue
        for ver in vers:
            found = sources.find_model(mid, ver, srcs, offline=offline)
            if not found:
                out.append((mid, ver, None))
                continue
            sha = found[2].get("artifact_sha256")
            ok = sha is None or _sha256_file(cache_path(mid, ver)) == sha
            out.append((mid, ver, ok))
    return out


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
