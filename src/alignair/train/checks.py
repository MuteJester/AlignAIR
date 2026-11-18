"""The checks training applies to its settings, each a ConfigurationError naming the setting."""
from __future__ import annotations

import math
from numbers import Real
from operator import index

import torch

from ..errors import ConfigurationError


def integer(value, field: str, minimum: int = 1) -> int:
    """``value`` as an int: an integer (any ``__index__``, not a bool) of at least ``minimum``."""
    try:
        if isinstance(value, bool):
            raise ValueError  # noqa: TRY004 - caught below: every refusal is one ConfigurationError
        value = index(value)
        if value < minimum:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be an integer >= {minimum}", context={"field": field}) from exc
    return value


def probability(value, field: str) -> float:
    """``value`` as a float: a finite number (not a bool) in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ConfigurationError(f"{field} must be a finite number in [0, 1]", context={"field": field})
    return float(value)


def device(value) -> torch.device:
    """``value`` as a torch device."""
    try:
        return torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError("invalid training device", context={"field": "device"}) from exc
