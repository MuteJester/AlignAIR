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

import json
import os
from urllib.parse import urljoin

DEFAULT_REGISTRY = "hf://alignair/alignair-models"
_HF = "https://huggingface.co"


class OfflineError(RuntimeError):
    """Raised when a fetch is attempted while offline."""


def _config_dir() -> str:
    """The per-OS user config dir (``platformdirs``: ~/.config on Linux honoring XDG_CONFIG_HOME,
    ~/Library/Application Support on macOS, %APPDATA% on Windows), with an XDG/home fallback."""
    if os.environ.get("ALIGNAIR_CONFIG_DIR"):
        return os.environ["ALIGNAIR_CONFIG_DIR"]
    try:
        import platformdirs
        return platformdirs.user_config_dir("alignair")           # already includes the app name
    except ImportError:                                            # pragma: no cover - stdlib fallback
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
        return os.path.join(base, "alignair")


def _config_registries(config_path: str | None) -> list[str]:
    path = config_path or os.path.join(_config_dir(), "config.toml")
    if not os.path.exists(path):
        return []
    import tomllib
    with open(path, "rb") as f:
        data = tomllib.load(f)
    regs = data.get("registries")
    return [str(r) for r in regs] if regs else []


def resolve_sources(cli: list[str] | None = None, *, config_path: str | None = None) -> list[str]:
    """Registry sources in precedence order (first non-empty wins)."""
    if cli:
        return list(cli)
    env = os.environ.get("ALIGNAIR_REGISTRY")
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    cfg = _config_registries(config_path)
    if cfg:
        return cfg
    return [DEFAULT_REGISTRY]


def _base(source: str) -> str:
    base = source
    if base.endswith("registry.json"):
        base = base[: -len("registry.json")]
    return base if base.endswith("/") else base + "/"


def artifact_url(source: str, relpath: str, *, revision: str = "main") -> str:
    """Absolute URL/path of ``relpath`` under ``source`` (registry.json or an artifact file)."""
    if source.startswith("hf://"):
        repo = source[len("hf://"):]
        return f"{_HF}/{repo}/resolve/{revision}/{relpath}"
    base = _base(source)
    if base.startswith("file://"):
        return base + relpath
    return urljoin(base, relpath)


def _local_path(url: str) -> str | None:
    if url.startswith("file://"):
        return url[len("file://"):]
    if "://" not in url:
        return url
    return None


def fetch_bytes(source: str, relpath: str, *, offline: bool = False, revision: str = "main") -> bytes:
    """Fetch a small file (e.g. ``registry.json``) from a source. Large artifacts stream via the
    cache module. Raises :class:`OfflineError` if ``offline`` and the source is remote."""
    url = artifact_url(source, relpath, revision=revision)
    local = _local_path(url)
    if local is not None:
        with open(local, "rb") as f:
            return f.read()
    if offline:
        raise OfflineError(f"offline: cannot fetch {url}")
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as r:            # nosec - user-configured registry
        return r.read()


def load_registry(source: str, *, offline: bool = False) -> dict:
    return json.loads(fetch_bytes(source, "registry.json", offline=offline).decode("utf-8"))


def find_model(model_id: str, version: str | None, sources: list[str], *, offline: bool = False):
    """Search ``sources`` in order for ``model_id`` (``version`` or the registry's ``latest``).
    Returns ``(source, version, version_entry, model_entry)`` or None. Unreachable sources are skipped."""
    for src in sources:
        try:
            reg = load_registry(src, offline=offline)
        except Exception:
            continue
        model = reg.get("models", {}).get(model_id)
        if not model:
            continue
        ver = version or model.get("latest")
        entry = model.get("versions", {}).get(ver)
        if entry is not None:
            return src, ver, entry, model
    return None
