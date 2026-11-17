"""Passive, careful "a newer model is available" check.

Suppressed entirely when offline / quiet / ``ALIGNAIR_NO_NETWORK`` / a pinned ``id@version`` was
used / no registry is reachable. The registry is fetched at most once per 24h (cached on disk);
notices go to stderr only and never raise — a routine ``predict`` is never delayed or failed by this.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from typing import TextIO

from . import sources
from .cache import cache_root, installed_models, version_key


def _registry_cache_path(src: str):
    h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return cache_root() / "registry" / f"{h}.json"


def cached_registry(src: str, *, ttl: int = 86400, offline: bool = False) -> dict | None:
    """Registry JSON for ``src``, fetched at most once per ``ttl`` seconds. Falls back to the last
    cached copy on any network error or when offline. Returns None if never fetched."""
    p = _registry_cache_path(src)
    fresh = p.exists() and (time.time() - p.stat().st_mtime) < ttl
    if fresh or offline:
        return json.loads(p.read_text()) if p.exists() else None
    try:
        reg = sources.load_registry(src)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(reg))
        return reg
    except Exception:  # noqa: BLE001 - the update check is best effort: fall back to the cached copy
        return json.loads(p.read_text()) if p.exists() else None


def maybe_notify_updates(*, registry: str | Iterable[str] | None = None, offline: bool = False, quiet: bool = False,
                         pinned: bool = False, stream: TextIO | None = None) -> list[tuple[str, str]]:
    """Print (to ``stream``, default stderr) one line per installed model that has a newer ``latest`` in the first
    reachable registry of ``registry`` (a string is one registry; default: the configured ones). Returns the
    (id, version) notices. Suppressed and swallows all errors per the conditions above."""
    stream = stream or sys.stderr
    if offline or quiet or pinned or os.environ.get("ALIGNAIR_NO_NETWORK"):
        return []
    try:
        installed = installed_models()
        if not installed:
            return []
        notices: list[tuple[str, str]] = []
        for src in sources.resolve_sources(registry):
            reg = cached_registry(src)
            if not reg:
                continue
            models = reg.get("models", {})
            for mid, vers in installed.items():
                latest = models.get(mid, {}).get("latest")
                if not latest or latest in vers:
                    continue
                if version_key(latest) > version_key(max(vers, key=version_key)):
                    trained = models[mid].get("versions", {}).get(latest, {}).get("trained", "")
                    when = f" ({trained})" if trained else ""
                    print(f"ℹ {mid} {latest}{when} available — run: alignair models update {mid}",
                          file=stream)
                    notices.append((mid, latest))
            break                                              # first reachable registry decides
        return notices
    except Exception:  # noqa: BLE001 - the update notice never fails a command
        return []
