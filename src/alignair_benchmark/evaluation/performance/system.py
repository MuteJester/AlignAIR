from __future__ import annotations

import platform

try:  # pragma: no cover - unavailable only on non-Unix platforms.
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


def current_rss_mb() -> float | None:
    """Return the process high-water RSS in MB when the platform exposes it."""

    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # pragma: no cover - defensive for unusual platforms.
        return None
    if usage <= 0:
        return None
    if platform.system() == "Darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0
