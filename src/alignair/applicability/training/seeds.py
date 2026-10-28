"""Deterministic production-seed selection using development metrics only."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

from alignair.errors import ConfigurationError, InputError

_RULE = "min_dev_loss_then_lowest_seed"
_FRESH_FIELDS = ("fresh", "selection", "loco", "blind", "far_ood", "far-ood", "seal", "locked", "test")


def choose_production_seed(
    dev_results: Mapping[int, Mapping[str, float]], frozen_rule: str = _RULE,
) -> int:
    """Choose the lowest development loss, breaking ties with the lowest seed.

    Fresh-validation fields are refused even if the winning seed would be the
    same. Missing/non-finite losses and unknown rules raise structured errors
    that remain catchable as ``ValueError``.
    """
    if frozen_rule != _RULE:
        raise ConfigurationError(
            "Unsupported production-seed selection rule.",
            context={"rule": frozen_rule, "supported_rule": _RULE},
        )
    if not isinstance(dev_results, Mapping) or not dev_results:
        raise InputError("Production-seed selection requires development results.")
    for seed, metrics in dev_results.items():
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise InputError("Production seeds must be integers.", context={"seed": seed})
        if not isinstance(metrics, Mapping):
            raise InputError("Development metrics must be a mapping.", context={"seed": seed})
        for key in metrics:
            if any(field in str(key).lower() for field in _FRESH_FIELDS):
                raise InputError(
                    f"choose_production_seed received a fresh-validation field: {key!r}",
                    context={"seed": seed, "field": key},
                )
        loss = metrics.get("dev_loss")
        if isinstance(loss, bool) or not isinstance(loss, Real) or not math.isfinite(loss):
            raise InputError(
                "Development loss must be a finite number.", context={"seed": seed},
            )
    return int(min(dev_results, key=lambda seed: (dev_results[seed]["dev_loss"], seed)))
