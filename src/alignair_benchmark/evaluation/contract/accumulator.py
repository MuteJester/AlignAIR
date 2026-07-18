from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.schema import BenchmarkCase
from .validators import validate_prediction, _case_has_d


@dataclass
class PredictionValidationAccumulator:
    """Streaming summary of prediction-contract validation."""

    level: str = "core"
    has_d: bool | None = True
    keep_rows: bool = False
    n_predictions: int = 0
    n_valid: int = 0
    coverage_sum: float = 0.0
    n_has_d: int = 0
    n_no_d: int = 0
    missing_field_counts: dict[str, int] = field(default_factory=dict)
    malformed_field_counts: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def _update_row(self, row: dict[str, Any]) -> None:
        self.n_predictions += 1
        self.n_valid += int(row["valid"])
        self.coverage_sum += float(row["coverage_fraction"])
        for field_name in row["missing_fields"]:
            self.missing_field_counts[field_name] = self.missing_field_counts.get(field_name, 0) + 1
        for item in row["malformed_fields"]:
            field_name = item["field"]
            self.malformed_field_counts[field_name] = self.malformed_field_counts.get(field_name, 0) + 1
        if self.keep_rows:
            self.rows.append(row)

    def update(self, predictions: list[dict[str, Any] | None], *, has_d: bool | None = None) -> None:
        resolved_has_d = self.has_d if has_d is None else has_d
        if resolved_has_d is None:
            raise ValueError("has_d must be provided when accumulator was initialized with has_d=None")
        for pred in predictions:
            row = validate_prediction(pred, level=self.level, has_d=resolved_has_d)
            self.n_has_d += int(resolved_has_d)
            self.n_no_d += int(not resolved_has_d)
            self._update_row(row)

    def update_for_cases(
        self,
        cases: list[BenchmarkCase],
        predictions: list[dict[str, Any] | None],
    ) -> None:
        if len(cases) != len(predictions):
            raise ValueError(f"case/prediction length mismatch: {len(cases)} != {len(predictions)}")
        for case, pred in zip(cases, predictions):
            has_d = _case_has_d(case)
            row = validate_prediction(pred, level=self.level, has_d=has_d)
            self.n_has_d += int(has_d)
            self.n_no_d += int(not has_d)
            self._update_row(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "has_d": self.has_d,
            "case_aware": self.has_d is None,
            "has_d_counts": {"has_d": self.n_has_d, "no_d": self.n_no_d},
            "n_predictions": self.n_predictions,
            "n_valid": self.n_valid,
            "valid_fraction": (self.n_valid / self.n_predictions) if self.n_predictions else 1.0,
            "mean_coverage_fraction": (
                self.coverage_sum / self.n_predictions if self.n_predictions else 1.0
            ),
            "missing_field_counts": dict(sorted(self.missing_field_counts.items())),
            "malformed_field_counts": dict(sorted(self.malformed_field_counts.items())),
            "rows": self.rows if self.keep_rows else None,
        }
