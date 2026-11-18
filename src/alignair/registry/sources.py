"""Registry source resolution + fetch.

A *registry source* points at the directory that holds ``registry.json`` and the ``.alignair``
artifacts it references. Three schemes, all fetchable with the stdlib:

  * ``hf://<org>/<repo>``  -> ``https://huggingface.co/<repo>/resolve/<rev>/<path>`` (public);
    ``huggingface_hub`` is lazy-imported ONLY for private repos / resumable large downloads.
  * ``https://<host>/.../[registry.json]``  -> a self-hosted mirror (trailing ``registry.json`` is
    stripped to get the base).
  * ``file://<path>``  -> a local / private-lab registry.

Resolution precedence: explicit CLI > ``ALIGNAIR_REGISTRY`` (comma-separated — URLs contain colons,
so never colon-split) > ``~/.config/alignair/config.toml`` ``registries`` > the default HF namespace.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import re
import urllib.error
from collections.abc import Iterable
from urllib.parse import urljoin

from alignair.errors import (
    OfflineError,
    RegistryError,
    RegistryInvalidError,
    RegistryUnavailableError,
    error_context,
)

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

DEFAULT_REGISTRY = "hf://AlignAIR/AlignAIR-pretrained"
_HF = "https://huggingface.co"


# What a remote fetch raises when the network or the server fails (URLError includes an HTTP error answer).
NETWORK_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException)


def retryable_status(status: int | None) -> bool:
    """Whether a retry may succeed after an HTTP answer ``status`` (None: no answer): not after a 4xx, which says the
    request itself is wrong, but 408 (timeout) and 429 (rate limit)."""
    return status is None or not 400 <= status < 500 or status in (408, 429)


@contextlib.contextmanager
def network_errors(url: str):
    """Raise a network failure inside the block (``NETWORK_ERRORS``) as a RegistryUnavailableError naming ``url``,
    retryable unless the server answered that the request is wrong. Other errors (a local file, the disk) pass."""
    try:
        yield
    except NETWORK_ERRORS as exc:
        status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        raise RegistryUnavailableError(f"could not fetch {url}: {exc}", context={"url": url, "http_status": status},
                                       hint="Check connectivity and the registry source.",
                                       retryable=retryable_status(status)) from exc


def config_dir() -> str:
    """The per-OS user config dir (``platformdirs``: ~/.config on Linux honoring XDG_CONFIG_HOME,
    ~/Library/Application Support on macOS, %APPDATA% on Windows), or ``ALIGNAIR_CONFIG_DIR`` when set."""
    import platformdirs
    if os.environ.get("ALIGNAIR_CONFIG_DIR"):
        return os.environ["ALIGNAIR_CONFIG_DIR"]
    return platformdirs.user_config_dir("alignair")           # already includes the app name


def _config_registries(config_path: str | None) -> list[str]:
    path = config_path or os.path.join(config_dir(), "config.toml")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = tomllib.load(f)
    regs = data.get("registries")
    return [str(r) for r in registry_list(regs)] if regs else []


def registry_list(registry: str | Iterable[str]) -> list[str]:
    """``registry`` as a list of registry sources: a string is one registry."""
    return [registry] if isinstance(registry, str) else list(registry)


def resolve_sources(registry: str | Iterable[str] | None = None, *, config_path: str | None = None) -> list[str]:
    """Registry sources in precedence order (first non-empty wins): ``registry`` (the CLI's ``--registry``; a string
    is one registry), ``ALIGNAIR_REGISTRY``, the config file, then the built-in catalog."""
    registry = registry_list(registry or [])
    if registry:
        return registry
    env = os.environ.get("ALIGNAIR_REGISTRY")
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    cfg = _config_registries(config_path)
    if cfg:
        return cfg
    return [DEFAULT_REGISTRY]


def _base(source: str) -> str:
    base = source
    base = base.removesuffix("registry.json")
    return base if base.endswith("/") else base + "/"


def artifact_url(source: str, relpath: str, *, revision: str = "main") -> str:
    """Absolute URL/path of ``relpath`` under ``source`` (registry.json or an artifact file)."""
    if source.startswith("hf://"):
        repo = source[len("hf://"):]
        return f"{_HF}/{repo}/resolve/{revision}/{relpath}"
    base = _base(source)
    if base.startswith("file://"):
        return base + relpath
    if "://" not in base:          # a local path, joined as one: urljoin reads a Windows drive (C:) as a URL scheme
        return os.path.join(base, relpath)
    return urljoin(base, relpath)


# A Windows drive after a file URL's third slash (``file:///C:/reg``, RFC 8089's form of a Windows path).
_DRIVE_AFTER_SLASH = re.compile(r"/[A-Za-z]:(?=[/\\]|$)")


def local_path(url: str) -> str | None:
    """The filesystem path of a ``file://`` URL or a bare path; None for a remote URL. A Windows drive after the
    third slash (``file:///C:/reg``) loses the slash before it, as ``urllib``'s ``url2pathname`` drops it: ``C:/reg``.
    """
    if url.startswith("file://"):
        path = url[len("file://"):]
        return path[1:] if _DRIVE_AFTER_SLASH.match(path) else path
    if "://" not in url:
        return url
    return None


def open_artifact(source: str, relpath: str, *, offline: bool = False, revision: str = "main", timeout: float = 30):
    """A binary stream of ``relpath`` from ``source``: the file of a local source, else the remote one (``timeout``
    seconds to connect). Raises :class:`OfflineError` if ``offline`` and the source is remote."""
    url = artifact_url(source, relpath, revision=revision)
    local = local_path(url)
    if local is not None:
        return open(local, "rb")
    if offline:
        raise OfflineError(f"offline: cannot fetch {url}", context={"url": url, "offline": True})
    import urllib.request
    with network_errors(url):
        return urllib.request.urlopen(url, timeout=timeout)        # nosec - user-configured registry


def fetch_bytes(source: str, relpath: str, *, offline: bool = False, revision: str = "main") -> bytes:
    """A small file (e.g. ``registry.json``) of a source (``open_artifact``); the cache streams large artifacts."""
    with network_errors(artifact_url(source, relpath, revision=revision)), \
            open_artifact(source, relpath, offline=offline, revision=revision) as stream:
        return stream.read()


def load_registry(source: str, *, offline: bool = False) -> dict:
    """The ``registry.json`` catalog of ``source``, checked: RegistryUnavailableError when it cannot be fetched (an
    OSError for a local one that cannot be opened), RegistryInvalidError when it is not a catalog."""
    with error_context(source=source, stage="registry_lookup"):
        try:
            registry = json.loads(fetch_bytes(source, "registry.json", offline=offline).decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            if isinstance(exc, RegistryError):
                raise
            raise RegistryInvalidError("invalid registry JSON", hint="Check the configured registry source.") from exc
        if not isinstance(registry, dict) or not isinstance(registry.get("models"), dict):
            raise RegistryInvalidError("registry must contain a models mapping")
        return registry


def find_model(model_id: str, version: str | None, registry: str | Iterable[str], *,
               offline: bool = False) -> tuple[str, str, dict, dict] | None:
    """Search the registries of ``registry`` in order (a string is one registry) for ``model_id`` (``version`` or the
    registry's ``latest``). Returns ``(source, version, version_entry, model_entry)`` or None. Failed sources are
    tried in order; if no match is found, failures are reported rather than called 'unknown model': an OfflineError
    when every source failed for being offline, a RegistryUnavailableError when one could not be reached or read
    (retryable when one of those is), else a RegistryInvalidError."""
    failures = []
    for src in registry_list(registry):
        try:
            reg = load_registry(src, offline=offline)
        except (OSError, RegistryError) as exc:
            failures.append((src, exc))
            continue
        model = reg.get("models", {}).get(model_id)
        if not model:
            continue
        if not isinstance(model, dict) or not isinstance(model.get("versions"), dict):
            failures.append((src, RegistryInvalidError("invalid model registry entry")))
            continue
        ver = version or model.get("latest")
        entry = model.get("versions", {}).get(ver)
        if entry is not None and not (isinstance(entry, dict) and isinstance(entry.get("file"), str) and entry["file"]):
            failures.append((src, RegistryInvalidError("invalid model version entry: it names no artifact file",
                                                       context={"model_id": model_id, "version": ver})))
            continue
        if entry is not None:
            return src, ver, entry, model
    if failures:
        raise _lookup_failure(failures, context={"model_id": model_id, "version": version,
                                                 "failed_sources": [s for s, _ in failures]}) from failures[-1][1]
    return None


def _lookup_failure(failures, *, context) -> RegistryError:
    """The error a lookup whose sources failed raises (``find_model``)."""
    errors = [error for _, error in failures]
    hint = "Check registry connectivity, configuration, and offline/cache settings."
    if all(isinstance(error, OfflineError) for error in errors):
        return OfflineError("offline: model is not available in the local cache or registries",
                            context={**context, "offline": True}, hint=hint)
    unreachable = [error for error in errors if not isinstance(error, (OfflineError, RegistryInvalidError))]
    if unreachable:
        return RegistryUnavailableError("could not resolve model because registry sources failed", context=context,
                                        hint=hint, retryable=any(getattr(e, "retryable", False) for e in unreachable))
    return RegistryInvalidError("could not resolve model because a registry source is invalid", context=context,
                                hint=hint)
