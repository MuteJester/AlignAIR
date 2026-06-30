from __future__ import annotations

from typing import Any


def normalize_call_set(pred: dict[str, Any], gene: str) -> tuple[str, ...]:
    """Extract a predicted allele set for one gene from flexible prediction keys."""

    calls = pred.get(f"{gene}_calls")
    if calls is None:
        call = pred.get(f"{gene}_call")
        if call is None:
            return ()
        calls = call.split(",") if isinstance(call, str) else [call]
    if isinstance(calls, str):
        calls = calls.split(",")
    return tuple(str(c).strip() for c in calls if str(c).strip())
