"""Bounded diagnostic snapshots and task-local operation context (standard library only)."""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from enum import Enum
from pathlib import PurePath
from urllib.parse import urlsplit, urlunsplit

_CONTEXT: ContextVar[dict | None] = ContextVar("alignair_error_context", default=None)
_REDACTED = "[REDACTED]"
# The keys whose values a diagnostic never carries: credentials, and the read payloads (sequences are user data).
_REDACTED_KEYS = {"token", "password", "secret", "authorization", "cookie", "api_key",
                  "sequence", "sequences", "sequence_aa", "tokenized_sequence", "raw_sequence"}
# The bounds every diagnostic payload keeps to.
MAX_TEXT = 1024        # characters of one text value
MAX_ITEMS = 50         # entries of one mapping or list
MAX_DEPTH = 6          # levels of nesting


def _sensitive(key: str) -> bool:
    key = key.lower().replace("-", "_").replace(" ", "_")
    return (key in _REDACTED_KEYS or any(key.endswith("_" + item) for item in _REDACTED_KEYS)
            or key in {"apikey", "accesskey", "access_key", "credentials"})


def snapshot(context: Mapping) -> dict:
    """Copy context into bounded, strict-JSON values without inspecting arbitrary objects.

    Credentials and sequence fields are redacted by key; URL credentials, query strings,
    and fragments are removed. Messages and tracebacks are deliberately NOT processed here.
    """
    return _clean(context, 0, set())


def _clean(value, depth: int, seen: set[int]):
    if isinstance(value, Enum):
        return _clean(value.value, depth, seen)
    if isinstance(value, PurePath):
        value = str(value)
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return _clean_container(value, depth, seen)
    return _clean_scalar(value)


def _clean_scalar(value):
    """None, a bool or an int as it is; a float as it is when finite, else its text; any other object its type."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return f"<{type(value).__name__}>"


def _clean_text(value: str) -> str:
    """A URL without its credentials, query and fragment; any text cut to ``MAX_TEXT`` characters."""
    if value.startswith(("http://", "https://", "hf://")):
        try:
            url = urlsplit(value)
            value = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path,
                                _REDACTED if url.query else "", ""))
        except ValueError:
            return "[invalid URL]"
    return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "[truncated]"


def _clean_container(value, depth: int, seen: set[int]):
    """A mapping (sensitive keys redacted) or a list of its first ``MAX_ITEMS`` entries cleaned, down to
    ``MAX_DEPTH`` levels; a cycle is marked."""
    if depth >= MAX_DEPTH:
        return "[depth limit]"
    if id(value) in seen:
        return "[cycle]"
    seen.add(id(value))
    try:
        if isinstance(value, Mapping):
            out = {}
            for i, (key, item) in enumerate(value.items()):
                if i == MAX_ITEMS:
                    out["__truncated__"] = True
                    break
                key = key if isinstance(key, str) else f"<{type(key).__name__} key>"
                out[key] = _REDACTED if _sensitive(key) else _clean(item, depth + 1, seen)
            return out
        out = []
        for i, item in enumerate(value):
            if i == MAX_ITEMS:
                out.append("[truncated]")
                break
            out.append(_clean(item, depth + 1, seen))
        return out
    finally:
        seen.remove(id(value))


def current_error_context() -> dict:
    """Return an independent copy of the current operation's diagnostic context."""
    return deepcopy(_CONTEXT.get() or {})


@contextmanager
def error_context(**context: object) -> Iterator[None]:
    """Attach context to errors created inside this scope; nesting and async tasks are isolated.

    This neither catches exceptions nor changes their type. Explicit exception context
    takes precedence over the scoped context. No sequence payloads should be supplied.
    """
    token = _CONTEXT.set(snapshot({**current_error_context(), **context}))
    try:
        yield
    finally:
        _CONTEXT.reset(token)
