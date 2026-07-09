from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionMatchResult:
    """Predictions aligned to benchmark case order plus match diagnostics."""

    predictions: list[dict[str, Any]]
    report: dict[str, Any]
