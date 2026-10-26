"""Bounded diagnostic snapshots and task-local operation context (standard library only)."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from enum import Enum
import math
from pathlib import PurePath
from urllib.parse import urlsplit, urlunsplit

_CONTEXT: ContextVar[dict | None] = ContextVar("alignair_error_context", default=None)
_REDACTED = "[REDACTED]"
_SECRET_KEYS = {"token", "password", "secret", "authorization", "cookie", "api_key",
                "sequence", "sequences", "sequence_aa", "tokenized_sequence", "raw_sequence"}


def _sensitive(key: str) -> bool:
    key = key.lower().replace("-", "_").replace(" ", "_")
    return (key in _SECRET_KEYS or any(key.endswith("_" + item) for item in _SECRET_KEYS)
            or key in {"apikey", "accesskey", "access_key", "credentials"})


def snapshot(context: Mapping) -> dict:
    """Copy context into bounded, strict-JSON values without inspecting arbitrary objects.

    Credentials and sequence fields are redacted by key; URL credentials, query strings,
    and fragments are removed. Messages and tracebacks are deliberately NOT processed here.
    """
    seen: set[int] = set()

    def clean(value, depth=0):
        if isinstance(value, Enum):
            return clean(value.value, depth)
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, PurePath):
            value = str(value)
        if isinstance(value, str):
            if value.startswith(("http://", "https://", "hf://")):
                try:
                    url = urlsplit(value)
                    value = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path,
                                        _REDACTED if url.query else "", ""))
                except ValueError:
                    return "[invalid URL]"
            return value if len(value) <= 1024 else value[:1024] + "[truncated]"
        if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
            return f"<{type(value).__name__}>"
        if depth >= 6:
            return "[depth limit]"
        if id(value) in seen:
            return "[cycle]"
        seen.add(id(value))
        try:
            if isinstance(value, Mapping):
                out = {}
                for i, (key, item) in enumerate(value.items()):
                    if i == 50:
                        out["__truncated__"] = True
                        break
                    key = key if isinstance(key, str) else f"<{type(key).__name__} key>"
                    out[key] = _REDACTED if _sensitive(key) else clean(item, depth + 1)
                return out
            out = []
            for i, item in enumerate(value):
                if i == 50:
                    out.append("[truncated]")
                    break
                out.append(clean(item, depth + 1))
            return out
        finally:
            seen.remove(id(value))

    return clean(context)


def current_error_context() -> dict:
    """Return an independent copy of the current operation's diagnostic context."""
    return deepcopy(_CONTEXT.get() or {})


@contextmanager
def error_context(**context):
    """Attach context to errors created inside this scope; nesting and async tasks are isolated.

    This neither catches exceptions nor changes their type. Explicit exception context
    takes precedence over the scoped context. No sequence payloads should be supplied.
    """
    token = _CONTEXT.set(snapshot({**current_error_context(), **context}))
    try:
        yield
    finally:
        _CONTEXT.reset(token)
