"""Prediction contract for benchmark-compatible aligner outputs."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.schema import GENES

LEVELS = ("minimal", "core", "assay")


@dataclass(frozen=True)
class PredictionField:
    """One prediction field accepted by benchmark scorers."""

    name: str
    dtype: str
    level: str
    description: str
    required_if_has_d: bool = False
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gene_fields(name: str, dtype: str, level: str, description: str) -> tuple[PredictionField, ...]:
    return tuple(
        PredictionField(
            name=f"{gene}_{name}",
            dtype=dtype,
            level=level,
            description=description.format(gene=gene.upper()),
            required_if_has_d=(gene == "d"),
        )
        for gene in GENES
    )


PREDICTION_FIELDS: tuple[PredictionField, ...] = (
    PredictionField("sequence_id", "str", "assay", "Stable input sequence identifier."),
    PredictionField("sequence", "str", "assay", "Presented sequence used by the aligner."),
    PredictionField("locus", "str", "assay", "Locus or chain label, for example IGH."),
    PredictionField("orientation_id", "int[0..3]", "core", "Presented-read orientation class."),
    *_gene_fields("call", "str", "minimal", "Top {gene} allele call."),
    *_gene_fields("calls", "list[str]", "core", "Set-valued {gene} allele call."),
    *_gene_fields("sequence_start", "int", "core", "{gene} start in the evaluated query frame."),
    *_gene_fields("sequence_end", "int", "core", "{gene} end in the evaluated query frame."),
    *_gene_fields("germline_start", "int", "core", "{gene} start in the called germline allele."),
    *_gene_fields("germline_end", "int", "core", "{gene} end in the called germline allele."),
    *_gene_fields("cigar", "str", "assay", "{gene} alignment CIGAR."),
    *_gene_fields("identity", "float", "assay", "{gene} identity/supporting alignment quality."),
    *_gene_fields("trim_5", "int", "assay", "{gene} 5-prime trimming."),
    *_gene_fields("trim_3", "int", "assay", "{gene} 3-prime trimming."),
    *_gene_fields("ranked_calls", "list[str]", "assay", "Ranked {gene} candidate list."),
    PredictionField("productive", "bool|float", "core", "Predicted productivity status."),
    PredictionField("vj_in_frame", "bool", "assay", "Whether V/J are in frame."),
    PredictionField("stop_codon", "bool", "assay", "Whether the translated sequence contains a stop codon."),
    PredictionField("junction", "str", "assay", "Junction/CDR3 nucleotide sequence."),
    PredictionField("junction_aa", "str", "assay", "Junction/CDR3 amino-acid sequence."),
    PredictionField("junction_start", "int", "assay", "Junction/CDR3 start in the evaluated query frame."),
    PredictionField("junction_end", "int", "assay", "Junction/CDR3 end in the evaluated query frame."),
    PredictionField("junction_length", "int", "assay", "Junction/CDR3 nucleotide length."),
    PredictionField("np1", "str", "assay", "N/P sequence between V and D."),
    PredictionField("np2", "str", "assay", "N/P sequence between D and J."),
    PredictionField("np1_length", "int", "assay", "Length of the V-D N/P region."),
    PredictionField("np2_length", "int", "assay", "Length of the D-J N/P region."),
    PredictionField("p_region_length", "int", "assay", "Total P-nucleotide length when reported directly."),
    PredictionField("d_inverted", "bool", "assay", "Whether D was called in inverted orientation."),
    PredictionField("c_call", "str", "assay", "Constant-region allele/gene call when applicable."),
    PredictionField("region_labels", "list[int]", "assay", "Per-position region labels in benchmark encoding."),
    PredictionField("state_labels", "list[int]", "assay", "Per-position edit-state labels in benchmark encoding."),
    PredictionField("mutation_rate", "float", "assay", "Predicted SHM mutation rate."),
    PredictionField("noise_count", "float", "assay", "Predicted sequencing/PCR noise count."),
    PredictionField("indel_count", "float", "assay", "Predicted indel burden."),
    PredictionField("read_layout", "str", "assay", "Read-layout label, for example paired_end."),
    PredictionField("is_contaminant", "bool", "assay", "Whether the input is a contaminant/out-of-scope read."),
    PredictionField("receptor_revision_applied", "bool", "assay", "Whether receptor revision was detected."),
)


def prediction_contract() -> list[dict[str, Any]]:
    """Return the benchmark prediction field contract as serializable data."""

    return [field.to_dict() for field in PREDICTION_FIELDS]


def _expected_fields(level: str, has_d: bool) -> tuple[PredictionField, ...]:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    max_idx = LEVELS.index(level)
    fields = []
    for field in PREDICTION_FIELDS:
        if LEVELS.index(field.level) > max_idx:
            continue
        if field.name.startswith("d_") and not has_d:
            continue
        fields.append(field)
    return tuple(fields)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool) or _is_missing(value):
        return False
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _is_float_like(value: Any) -> bool:
    if isinstance(value, bool) or _is_missing(value):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and value in (0, 1):
        return True
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "t", "f", "yes", "no", "1", "0"}:
        return True
    return False


def _matches_dtype(value: Any, dtype: str) -> bool:
    if _is_missing(value):
        return True
    if dtype == "str":
        return isinstance(value, str)
    if dtype == "list[str]":
        return isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
    if dtype == "list[int]":
        return isinstance(value, (list, tuple)) and all(_is_int_like(v) for v in value)
    if dtype == "int":
        return _is_int_like(value)
    if dtype == "float":
        return _is_float_like(value)
    if dtype == "bool":
        return _is_bool_like(value)
    if dtype == "bool|float":
        return _is_bool_like(value) or _is_float_like(value)
    if dtype == "int[0..3]":
        return _is_int_like(value) and 0 <= int(float(value)) <= 3
    return True


def validate_prediction(
    prediction: dict[str, Any] | None,
    *,
    level: str = "core",
    has_d: bool = True,
) -> dict[str, Any]:
    """Validate one normalized prediction dictionary against a contract level."""

    pred = prediction or {}
    fields = _expected_fields(level, has_d)
    expected_names = [field.name for field in fields]
    missing = [field.name for field in fields if _is_missing(pred.get(field.name))]
    malformed = []
    for field in fields:
        if field.name in pred and not _matches_dtype(pred[field.name], field.dtype):
            malformed.append({"field": field.name, "expected": field.dtype, "value_type": type(pred[field.name]).__name__})
    present = [name for name in expected_names if name not in missing]
    return {
        "valid": not missing and not malformed,
        "level": level,
        "has_d": has_d,
        "n_expected": len(expected_names),
        "n_present": len(present),
        "coverage_fraction": len(present) / len(expected_names) if expected_names else 1.0,
        "present_fields": present,
        "missing_fields": missing,
        "malformed_fields": malformed,
        "extra_fields": sorted(k for k in pred if k not in {field.name for field in PREDICTION_FIELDS}),
    }


def validate_predictions(
    predictions: list[dict[str, Any] | None],
    *,
    level: str = "core",
    has_d: bool = True,
) -> dict[str, Any]:
    """Validate a batch of normalized prediction dictionaries."""

    rows = [validate_prediction(pred, level=level, has_d=has_d) for pred in predictions]
    missing_counts: dict[str, int] = {}
    malformed_counts: dict[str, int] = {}
    for row in rows:
        for field in row["missing_fields"]:
            missing_counts[field] = missing_counts.get(field, 0) + 1
        for item in row["malformed_fields"]:
            field = item["field"]
            malformed_counts[field] = malformed_counts.get(field, 0) + 1
    return {
        "level": level,
        "has_d": has_d,
        "n_predictions": len(rows),
        "n_valid": sum(1 for row in rows if row["valid"]),
        "valid_fraction": (sum(1 for row in rows if row["valid"]) / len(rows)) if rows else 1.0,
        "mean_coverage_fraction": (
            sum(row["coverage_fraction"] for row in rows) / len(rows)
            if rows
            else 1.0
        ),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "malformed_field_counts": dict(sorted(malformed_counts.items())),
        "rows": rows,
    }


@dataclass
class PredictionValidationAccumulator:
    """Streaming summary of prediction-contract validation."""

    level: str = "core"
    has_d: bool = True
    keep_rows: bool = False
    n_predictions: int = 0
    n_valid: int = 0
    coverage_sum: float = 0.0
    missing_field_counts: dict[str, int] = field(default_factory=dict)
    malformed_field_counts: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def update(self, predictions: list[dict[str, Any] | None]) -> None:
        for pred in predictions:
            row = validate_prediction(pred, level=self.level, has_d=self.has_d)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "has_d": self.has_d,
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
