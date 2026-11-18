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
from collections.abc import Iterable
from pathlib import Path

from alignair.errors import DownloadIntegrityError, ModelError, ModelNotFoundError, RegistryError

from ..io.files import sha256_file
from . import sources as registry_sources


def cache_root() -> Path:
    """The local model cache root. Honors ``ALIGNAIR_CACHE_DIR``; otherwise the correct per-OS user
    cache dir via ``platformdirs`` (``~/.cache/alignair`` on Linux honoring ``XDG_CACHE_HOME``,
    ``~/Library/Caches/alignair`` on macOS, ``%LOCALAPPDATA%\\alignair`` on Windows)."""
    import platformdirs
    if os.environ.get("ALIGNAIR_CACHE_DIR"):
        return Path(os.environ["ALIGNAIR_CACHE_DIR"])
    return Path(platformdirs.user_cache_dir("alignair"))


def cache_path(model_id: str, version: str) -> Path:
    """Where version ``version`` of ``model_id`` is cached."""
    return cache_root() / "models" / model_id / f"{version}.alignair"


def _sidecar(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def _write_sidecar(dest: Path, sha: str | None) -> None:
    """Record the verified artifact hash next to it, so a later offline pinned-cache hit can prove the
    file was not corrupted/swapped after download."""
    if sha:
        try:
            _sidecar(dest).write_text(sha)
        except OSError:
            pass


def _sidecar_ok(path: Path):
    """True/False if a sidecar exists (matches / mismatches), None if there is no sidecar to check."""
    sc = _sidecar(path)
    if not sc.exists():
        return None
    try:
        expected = sc.read_text().strip()
    except OSError:
        return None
    return sha256_file(path) == expected


@contextlib.contextmanager
def _excl_lock(path: Path, *, timeout: float = 600.0, stale_after: float = 3600.0, poll: float = 0.05):
    """Portable cross-process mutual exclusion via an atomic ``O_EXCL`` lockfile (works where ``flock``
    is absent, e.g. Windows, so concurrent downloads cannot corrupt the cache).
    Acquire by exclusive create; spin-wait while another holder has it; reclaim a *stale* lock (a dead
    holder's file older than ``stale_after``) so a crash can't deadlock the cache; time out otherwise."""
    import time
    fd = None
    permission_miss = False
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            permission_miss = False
            # another holder owns the lockfile: contention
        except PermissionError:
            # Windows raises PermissionError (a sharing violation: WinError 32/5 -> errno 13) instead
            # of FileExistsError when another holder has the lockfile open. Treat that as contention,
            # but only while the lockfile exists. The holder may release between os.open() failing
            # and this existence check, so retry that disappearing-lock race once. A persistent
            # permission failure with no lockfile still propagates promptly rather than timing out.
            if not os.path.exists(path):
                if permission_miss:
                    raise
                permission_miss = True
                continue
            permission_miss = False
        # contention: reclaim a stale lock (a dead holder's file), else wait until the deadline.
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
    # The lock is represented by the atomically-created file, not by an open handle. Closing the
    # handle before the critical section avoids Windows sharing violations for other contenders;
    # they can then observe the existing file through the normal FileExistsError path.
    try:
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
            fd = None
        yield
    finally:
        if fd is not None:                         # acquisition succeeded but initialization failed
            os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass


@contextlib.contextmanager
def cache_lock(path: Path):
    """Hold a cross-process lock on ``path`` (``flock``, or a lockfile where there is none)."""
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


def download_verified(source: str, relpath: str, dest: Path, expected_sha256: str | None = None, *,
                      offline: bool = False) -> Path:
    """Stream ``relpath`` from ``source`` into ``dest``, SHA256-verify, atomically install. Idempotent:
    a present, matching ``dest`` is returned without re-downloading. A connection that fails or drops is a retryable
    RegistryUnavailableError; a disk error stays an OSError."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    with cache_lock(dest.with_name(dest.name + ".lock")):
        if dest.exists() and (expected_sha256 is None or sha256_file(dest) == expected_sha256):
            _write_sidecar(dest, expected_sha256)                 # record hash for later offline checks
            return dest
        h = hashlib.sha256()
        with registry_sources.network_errors(registry_sources.artifact_url(source, relpath)), \
                registry_sources.open_artifact(source, relpath, offline=offline, timeout=60) as r, \
                open(part, "wb") as w:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                w.write(chunk)
        got = h.hexdigest()
        if expected_sha256 and got != expected_sha256:
            part.unlink(missing_ok=True)
            raise DownloadIntegrityError(
                f"SHA256 mismatch for {relpath}: expected {expected_sha256}, got {got} — refusing to install.",
                context={"stage": "model_download", "source": source, "artifact": relpath,
                         "path": dest, "expected_sha256": expected_sha256, "actual_sha256": got},
                hint="Re-download from a trusted source; do not bypass integrity verification.")
        os.replace(part, dest)                                     # atomic
        _write_sidecar(dest, expected_sha256 or got)
    return dest


def resolve_model(spec: str | os.PathLike[str], *, registry: str | Iterable[str] | None = None,
                  offline: bool = False, token: str | None = None, revision: str | None = None) -> Path:
    """Resolve a model spec to a local artifact path:

    * a filesystem path -> that path;
    * a Hugging Face repo (``hf://org/repo`` or ``org/repo``) -> the ``.alignair`` pulled straight from
      that repo via ``huggingface_hub`` (revision/token/offline honored — the one-repo-per-model path);
    * a catalog ``id`` / ``id@version`` -> the verified cached artifact from the first registry of ``registry`` (a
      string is one registry; default: the configured ones, ``resolve_sources``) that lists it.

    ``revision`` supplies a catalog version when no inline ``@version`` is present; for Hub repos
    it overrides the inline revision. Local paths ignore it.
    A pinned + already-cached version returns immediately (no network).
    """
    explicit_path = isinstance(spec, os.PathLike)
    spec = _spec_text(spec)
    local = _local_file(spec, explicit_path)
    if local is not None:
        return local
    from . import hf
    if hf.is_hf_repo_spec(spec):  # direct one-repo-per-model HF loading
        return hf.download_from_hub(spec, revision=revision, token=token, offline=offline)
    model_id, _, version = spec.partition("@")
    version = version or revision or None
    srcs = registry_sources.registry_list(registry) if registry is not None else registry_sources.resolve_sources()
    cached = _cached_without_lookup(model_id, version, offline)
    if cached is not None:
        return cached
    return _from_registry(spec, model_id, version, srcs, offline)


def _spec_text(spec) -> str:
    """``spec`` as a non-empty string; a ModelError otherwise."""
    try:
        spec = os.fspath(spec)
    except TypeError as exc:
        raise ModelError("model must be a path or non-empty model specification",
                         context={"stage": "model_resolution"}) from exc
    if not isinstance(spec, str) or not spec.strip():
        raise ModelError("model must be a path or non-empty model specification",
                         context={"stage": "model_resolution"})
    return spec


def _local_file(spec: str, explicit_path: bool) -> Path | None:
    """The model file ``spec`` names when it exists; ModelNotFoundError when it names a file (a path object, an absolute
    or relative path, a model file name) that does not exist, a ModelError when it is not a file; None for a repo or
    catalog id."""
    local = Path(spec).expanduser()
    if local.exists():
        if not local.is_file():
            raise ModelError("model path must be a file",
                             context={"stage": "model_resolution", "path": local})
        return local
    from . import hf
    if (explicit_path or os.path.isabs(spec) or spec.startswith(("./", "../", "~"))
            or not spec.startswith("hf://") and hf.names_model_file(spec)):
        raise ModelNotFoundError("model file does not exist",
                                 context={"stage": "model_resolution", "path": local})
    return None


def _cached_without_lookup(model_id: str, version: str | None, offline: bool) -> Path | None:
    """The cached artifact returned without a registry lookup: a pinned version already cached (pinned + cached -> no
    network), or offline, the latest cached version; else None."""
    if version:
        cached = cache_path(model_id, version)
        # return a cached pinned version without the network ONLY if its integrity sidecar still checks
        # out (or there is no sidecar to check: a cache written without one). A tampered/corrupt file falls through
        # to a registry re-resolve instead of loading the wrong model.
        return cached if cached.exists() and _sidecar_ok(cached) is not False else None
    return _latest_cached(model_id) if offline else None


def _from_registry(spec: str, model_id: str, version: str | None, srcs: list[str], offline: bool) -> Path:
    """The verified artifact of ``model_id`` (``version`` or the latest) from the first registry in ``srcs`` that
    lists it, downloaded unless already cached; when none does, the latest cached version of an unpinned id, else
    the lookup's error or a ModelNotFoundError naming the known ids."""
    lookup_error = None
    try:
        found = registry_sources.find_model(model_id, version, srcs, offline=offline)
    except RegistryError as exc:
        lookup_error = exc
        found = None

    if not found:
        cached = None if version else _latest_cached(model_id)
        if cached is not None:
            return cached
        if lookup_error is not None:
            raise lookup_error
        known = _known_ids(srcs, offline)
        hint = f" Known ids: {', '.join(known)}." if known else ""
        raise ModelNotFoundError(f"unknown model '{spec}' — pass a file path or a registry id.{hint}",
                                 context={"stage": "model_resolution", "model_id": model_id, "version": version},
                                 hint="Check the model ID, requested version, and registry configuration.")
    src, ver, entry, _ = found
    dest = cache_path(model_id, ver)
    sha = entry.get("artifact_sha256")
    if dest.exists() and (sha is None or sha256_file(dest) == sha):
        return dest
    return download_verified(src, entry["file"], dest, sha, offline=offline)


def _latest_cached(model_id: str) -> Path | None:
    """The latest cached version of ``model_id`` when its integrity sidecar checks out (or it has none: a cache
    written without one); else None."""
    inst = installed_models().get(model_id)
    if inst:
        cached = cache_path(model_id, inst[-1])
        if cached.exists() and _sidecar_ok(cached) is not False:
            return cached
    return None


def installed_models() -> dict[str, list[str]]:
    """{model_id: [installed versions]} scanned from the cache dir."""
    root = cache_root() / "models"
    out: dict[str, list[str]] = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir():
            vers = sorted((p.name[: -len(".alignair")] for p in d.glob("*.alignair")), key=version_key)
            if vers:
                out[d.name] = vers
    return out


def version_key(v: str):
    """A sort key for version strings: dotted integers numerically, before any other string."""
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


def verify_installed(registry: str | Iterable[str], *, offline: bool = False,
                     model_id: str | None = None) -> list[tuple[str, str, bool | None]]:
    """Re-hash each installed artifact against the registry: a list of (id, version, ok, or None when the registry
    does not know it)."""
    registry = registry_sources.registry_list(registry)       # read once per version below: not a one-shot iterator
    out = []
    for mid, vers in installed_models().items():
        if model_id and mid != model_id:
            continue
        for ver in vers:
            found = registry_sources.find_model(mid, ver, registry, offline=offline)
            if not found:
                out.append((mid, ver, None))
                continue
            sha = found[2].get("artifact_sha256")
            ok = sha is None or sha256_file(cache_path(mid, ver)) == sha
            out.append((mid, ver, ok))
    return out


def _known_ids(srcs, offline) -> list[str]:
    ids: list[str] = []
    for src in srcs:
        try:
            ids += list(registry_sources.load_registry(src, offline=offline).get("models", {}))
        except Exception:  # noqa: BLE001, S112 - the ids an error message suggests; an unreadable registry has none
            continue
    return sorted(set(ids))
