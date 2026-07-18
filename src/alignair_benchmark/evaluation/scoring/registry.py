"""Registry of global scoring components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..performance import prediction_performance_metrics
from ...core.schema import BenchmarkCase
from .contract import score_airr_contract
from .junction import score_junction
from .labels import score_labels, score_scalar
from .manifest import (
    AIRR_CONTRACT_MANIFEST,
    INDEL_COUNT_MANIFEST,
    JUNCTION_MANIFEST,
    METADATA_MANIFEST,
    MUTATION_RATE_MANIFEST,
    NOISE_COUNT_MANIFEST,
    ORIENTATION_MANIFEST,
    PERFORMANCE_MANIFEST,
    PRODUCTIVE_MANIFEST,
    REGION_LABELS_MANIFEST,
    SEGMENT_ORDER_MANIFEST,
    STATE_LABELS_MANIFEST,
    ScoringComponentManifest,
)
from .metadata import score_metadata
from .orientation import score_orientation
from .structure import score_segment_order

GlobalScoreFn = Callable[[dict[str, Any], BenchmarkCase, str], dict[str, float]]


@dataclass(frozen=True)
class GlobalScoringComponent:
    """A named scorer that contributes global per-case metrics."""

    name: str
    score: GlobalScoreFn
    manifest: ScoringComponentManifest


def _orientation(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_orientation(pred, case)


def _airr_contract(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_airr_contract(pred, case)


def _segment_order(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_segment_order(pred, case, frame)


def _junction(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_junction(pred, case, frame)


def _metadata(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_metadata(pred, case)


def _performance(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return prediction_performance_metrics(pred)


def _region_labels(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_labels(pred, case, "region", frame)


def _state_labels(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_labels(pred, case, "state", frame)


def _noise_count(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_scalar(pred, case, "noise_count")


def _mutation_rate(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_scalar(pred, case, "mutation_rate")


def _indel_count(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_scalar(pred, case, "indel_count")


def _productive(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    return score_scalar(pred, case, "productive")


GLOBAL_SCORING_COMPONENTS: tuple[GlobalScoringComponent, ...] = (
    GlobalScoringComponent("orientation", _orientation, ORIENTATION_MANIFEST),
    GlobalScoringComponent("airr_contract", _airr_contract, AIRR_CONTRACT_MANIFEST),
    GlobalScoringComponent("segment_order", _segment_order, SEGMENT_ORDER_MANIFEST),
    GlobalScoringComponent("junction", _junction, JUNCTION_MANIFEST),
    GlobalScoringComponent("metadata", _metadata, METADATA_MANIFEST),
    GlobalScoringComponent("performance", _performance, PERFORMANCE_MANIFEST),
    GlobalScoringComponent("region_labels", _region_labels, REGION_LABELS_MANIFEST),
    GlobalScoringComponent("state_labels", _state_labels, STATE_LABELS_MANIFEST),
    GlobalScoringComponent("noise_count", _noise_count, NOISE_COUNT_MANIFEST),
    GlobalScoringComponent("mutation_rate", _mutation_rate, MUTATION_RATE_MANIFEST),
    GlobalScoringComponent("indel_count", _indel_count, INDEL_COUNT_MANIFEST),
    GlobalScoringComponent("productive", _productive, PRODUCTIVE_MANIFEST),
)


def validate_global_scoring_registry(
    components: tuple[GlobalScoringComponent, ...] = GLOBAL_SCORING_COMPONENTS,
) -> list[str]:
    """Return configuration errors for the global scoring component registry."""

    errors: list[str] = []
    seen: set[str] = set()
    for component in components:
        if not component.name:
            errors.append("global scoring component has an empty name")
            continue
        if component.name in seen:
            errors.append(f"duplicate global scoring component: {component.name}")
        if component.manifest.name != component.name:
            errors.append(f"manifest name mismatch for global scoring component: {component.name}")
        if component.manifest.scope != "global":
            errors.append(f"global scoring component has non-global manifest scope: {component.name}")
        if not component.manifest.metric_keys:
            errors.append(f"global scoring component has no declared metrics: {component.name}")
        seen.add(component.name)
    return errors
