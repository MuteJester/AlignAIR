"""Versioned benchmark artifact contracts.

These contracts intentionally describe the current public artifact shapes. They
are lightweight guards for IO/report boundaries, not a replacement for the
domain-level scorers and readiness checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


CURRENT_SCHEMA_VERSION = "0.1"

BENCHMARK_CASE_JSONL = "benchmark_case_jsonl"
PREDICTION_JSONL = "prediction_jsonl"
BENCHMARK_REPORT = "benchmark_report"
BENCHMARK_MANIFEST = "benchmark_manifest"
BENCHMARK_SUITE_MANIFEST = "benchmark_suite_manifest"
BENCHMARK_READINESS_REPORT = "benchmark_readiness_report"
MODEL_COMPARISON_REPORT = "model_comparison_report"


@dataclass(frozen=True)
class ArtifactContract:
    """Minimal contract for one persisted benchmark artifact kind."""

    kind: str
    schema_version: str
    description: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    compatible_versions: tuple[str, ...] = field(default_factory=lambda: (CURRENT_SCHEMA_VERSION,))

    @property
    def schema_name(self) -> str:
        return f"alignair_benchmark.{self.kind}.v{self.schema_version}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_name"] = self.schema_name
        return data


ARTIFACT_CONTRACTS: dict[str, ArtifactContract] = {
    BENCHMARK_CASE_JSONL: ArtifactContract(
        kind=BENCHMARK_CASE_JSONL,
        schema_version=CURRENT_SCHEMA_VERSION,
        description="One JSON object per benchmark case.",
        required_fields=(
            "case_id",
            "stratum",
            "sequence",
            "canonical_sequence",
            "orientation_id",
            "genes",
            "presented_genes",
        ),
        optional_fields=(
            "region_labels",
            "state_labels",
            "presented_region_labels",
            "presented_state_labels",
            "scalars",
            "tags",
            "record",
        ),
    ),
    PREDICTION_JSONL: ArtifactContract(
        kind=PREDICTION_JSONL,
        schema_version=CURRENT_SCHEMA_VERSION,
        description="One normalized prediction dictionary per input sequence.",
        required_fields=(),
        optional_fields=(
            "sequence_id",
            "case_id",
            "sequence",
            "v_call",
            "d_call",
            "j_call",
            "v_calls",
            "d_calls",
            "j_calls",
        ),
    ),
    BENCHMARK_REPORT: ArtifactContract(
        kind=BENCHMARK_REPORT,
        schema_version=CURRENT_SCHEMA_VERSION,
        description="Full benchmark evaluation report.",
        required_fields=(
            "benchmark",
            "frame",
            "criteria",
            "prediction_contract",
            "scenario_axes",
            "coverage",
            "results",
            "diagnostics",
            "criteria_audit",
            "assay",
        ),
        optional_fields=(
            "artifact",
            "catalog_validation",
            "metric_registry",
            "scoring_manifest",
            "scoring_audit",
            "performance",
            "prediction_matching",
            "prediction_validation",
            "uncertainty",
        ),
    ),
    BENCHMARK_MANIFEST: ArtifactContract(
        kind=BENCHMARK_MANIFEST,
        schema_version=CURRENT_SCHEMA_VERSION,
        description="Export manifest for benchmark input artifacts.",
        required_fields=(
            "manifest_version",
            "benchmark",
            "generation",
            "coverage",
            "observed_truth",
            "coordinate_conventions",
            "files",
        ),
        optional_fields=(
            "artifact",
            "created_at_utc",
            "software",
            "reference",
            "measurement_coverage",
            "readiness",
        ),
    ),
    BENCHMARK_SUITE_MANIFEST: ArtifactContract(
        kind=BENCHMARK_SUITE_MANIFEST,
        schema_version=CURRENT_SCHEMA_VERSION,
        description="Export manifest for a composed benchmark suite pack.",
        required_fields=(
            "artifact",
            "manifest_version",
            "suite",
            "benchmark",
            "generation",
            "coverage",
            "measurement_coverage",
            "suite_readiness",
            "components",
            "files",
        ),
        optional_fields=(
            "created_at_utc",
            "software",
            "reference",
        ),
    ),
    BENCHMARK_READINESS_REPORT: ArtifactContract(
        kind=BENCHMARK_READINESS_REPORT,
        schema_version=CURRENT_SCHEMA_VERSION,
        description="Readiness assessment for generated benchmark case coverage.",
        required_fields=(
            "grade",
            "profile",
            "thresholds",
            "n_cases",
            "grade_counts",
            "checks",
            "coverage",
            "truth_source",
        ),
        optional_fields=(
            "artifact",
            "observed_required_context_counts",
        ),
    ),
    MODEL_COMPARISON_REPORT: ArtifactContract(
        kind=MODEL_COMPARISON_REPORT,
        schema_version=CURRENT_SCHEMA_VERSION,
        description="Paired model-vs-model benchmark comparison report.",
        required_fields=(
            "comparison",
            "summary",
            "overall",
            "by_stratum",
            "skipped_strata",
        ),
        optional_fields=(
            "artifact",
            "decision",
            "prediction_matching",
        ),
    ),
}


def artifact_contract(kind: str) -> ArtifactContract:
    """Return the contract for an artifact kind."""

    try:
        return ARTIFACT_CONTRACTS[kind]
    except KeyError as exc:
        known = ", ".join(sorted(ARTIFACT_CONTRACTS))
        raise ValueError(f"unknown benchmark artifact kind: {kind}; expected one of: {known}") from exc


def artifact_contract_catalog() -> list[dict[str, Any]]:
    """Return all artifact contracts as serializable dictionaries."""

    return [contract.to_dict() for contract in ARTIFACT_CONTRACTS.values()]


def artifact_metadata(kind: str) -> dict[str, Any]:
    """Return stable schema metadata for embedding in persisted artifacts."""

    contract = artifact_contract(kind)
    return {
        "kind": contract.kind,
        "schema_name": contract.schema_name,
        "schema_version": contract.schema_version,
    }


def validate_artifact(
    payload: Mapping[str, Any] | None,
    kind: str,
    *,
    require_current_version: bool = False,
) -> dict[str, Any]:
    """Validate top-level artifact shape and embedded version metadata.

    The return shape is intentionally report-like instead of exception-first so
    CLI tools and tests can show all contract problems at once.
    """

    contract = artifact_contract(kind)
    payload = payload or {}
    missing_fields = tuple(field for field in contract.required_fields if field not in payload)
    extra_version = None
    embedded_kind = None
    metadata = payload.get("artifact")
    if isinstance(metadata, Mapping):
        embedded_kind = metadata.get("kind")
        extra_version = metadata.get("schema_version")
    elif kind == BENCHMARK_MANIFEST:
        extra_version = payload.get("manifest_version")

    version_present = extra_version is not None
    compatible = (
        not version_present
        or str(extra_version) in contract.compatible_versions
        or (not require_current_version and str(extra_version) == contract.schema_version)
    )
    if require_current_version:
        compatible = str(extra_version) == contract.schema_version

    problems = []
    if missing_fields:
        problems.append("missing_required_fields")
    if embedded_kind is not None and embedded_kind != kind:
        problems.append("artifact_kind_mismatch")
    if version_present and not compatible:
        problems.append("incompatible_schema_version")

    return {
        "valid": not problems,
        "artifact_kind": kind,
        "schema_version": contract.schema_version,
        "schema_name": contract.schema_name,
        "embedded_artifact_kind": embedded_kind,
        "embedded_schema_version": extra_version,
        "version_present": version_present,
        "require_current_version": require_current_version,
        "missing_required_fields": missing_fields,
        "problems": tuple(problems),
    }
