"""Validation helpers for benchmark criteria and scenario catalogs."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .criteria import DEFAULT_CONTEXTS, criteria_catalog, scenario_axes_catalog
from .metric_registry import metric_registry


ALLOWED_CRITERION_STATUSES = ("available", "partial", "planned")
ALLOWED_CRITERION_IMPORTANCE = ("core", "diagnostic", "optional")


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    counts = Counter(values)
    return tuple(sorted(value for value, count in counts.items() if count > 1))


def _known_contexts(scenario_axes: Sequence[Mapping[str, Any]]) -> set[str]:
    contexts = set(DEFAULT_CONTEXTS)
    for axis in scenario_axes:
        name = axis.get("name")
        if isinstance(name, str) and name:
            contexts.add(name)
        contexts.update(str(value) for value in _as_sequence(axis.get("values")) if value)
    return contexts


def validate_catalogs(
    *,
    criteria: Sequence[Mapping[str, Any]] | None = None,
    scenario_axes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate static assay catalogs for structural consistency."""

    criteria_rows = tuple(criteria if criteria is not None else criteria_catalog())
    scenario_rows = tuple(scenario_axes if scenario_axes is not None else scenario_axes_catalog())
    registry_keys = set(metric_registry())
    known_contexts = _known_contexts(scenario_rows)

    criteria_names = [str(row.get("name", "")) for row in criteria_rows if row.get("name")]
    axis_names = [str(row.get("name", "")) for row in scenario_rows if row.get("name")]
    duplicate_criterion_names = _duplicates(criteria_names)
    duplicate_axis_names = _duplicates(axis_names)

    criterion_problems = []
    unmapped_contexts = set()
    metric_keys_without_registry = set()
    all_metric_keys = set()
    status_counts = Counter()
    importance_counts = Counter()
    for idx, row in enumerate(criteria_rows):
        name = str(row.get("name") or f"criteria[{idx}]")
        problems = []
        for field in ("category", "name", "description", "interpretation"):
            if not str(row.get(field, "")).strip():
                problems.append(f"missing_{field}")
        status = str(row.get("status", "available"))
        importance = str(row.get("importance", "core"))
        status_counts[status] += 1
        importance_counts[importance] += 1
        if status not in ALLOWED_CRITERION_STATUSES:
            problems.append("invalid_status")
        if importance not in ALLOWED_CRITERION_IMPORTANCE:
            problems.append("invalid_importance")

        metric_keys = tuple(str(key) for key in _as_sequence(row.get("metric_keys")) if key)
        if not metric_keys:
            problems.append("missing_metric_keys")
        duplicate_metric_keys = _duplicates(metric_keys)
        if duplicate_metric_keys:
            problems.append("duplicate_metric_keys")
        all_metric_keys.update(metric_keys)
        metric_keys_without_registry.update(key for key in metric_keys if key not in registry_keys)

        contexts = tuple(str(context) for context in _as_sequence(row.get("contexts")) if context)
        if not contexts:
            problems.append("missing_contexts")
        row_unmapped_contexts = tuple(context for context in contexts if context not in known_contexts)
        unmapped_contexts.update(row_unmapped_contexts)

        if problems or row_unmapped_contexts or duplicate_metric_keys:
            criterion_problems.append(
                {
                    "name": name,
                    "problems": tuple(problems),
                    "duplicate_metric_keys": duplicate_metric_keys,
                    "unmapped_contexts": row_unmapped_contexts,
                    "metric_keys_without_registry": tuple(
                        key for key in metric_keys if key not in registry_keys
                    ),
                }
            )

    axis_problems = []
    for idx, row in enumerate(scenario_rows):
        name = str(row.get("name") or f"scenario_axes[{idx}]")
        problems = []
        for field in ("name", "description", "why_it_matters"):
            if not str(row.get(field, "")).strip():
                problems.append(f"missing_{field}")
        values = tuple(str(value) for value in _as_sequence(row.get("values")) if value)
        if not values:
            problems.append("missing_values")
        duplicate_values = _duplicates(values)
        if duplicate_values:
            problems.append("duplicate_values")
        if problems:
            axis_problems.append(
                {
                    "name": name,
                    "problems": tuple(problems),
                    "duplicate_values": duplicate_values,
                }
            )

    problems = []
    if duplicate_criterion_names:
        problems.append("duplicate_criterion_names")
    if duplicate_axis_names:
        problems.append("duplicate_scenario_axis_names")
    if any(row["problems"] for row in criterion_problems):
        problems.append("invalid_criteria")
    if axis_problems:
        problems.append("invalid_scenario_axes")
    if metric_keys_without_registry:
        problems.append("criteria_metric_keys_without_registry")

    warnings = []
    if unmapped_contexts:
        warnings.append("criteria_contexts_without_scenario_axis")

    return {
        "valid": not problems,
        "problems": tuple(problems),
        "warnings": tuple(warnings),
        "summary": {
            "n_criteria": len(criteria_rows),
            "n_scenario_axes": len(scenario_rows),
            "n_metric_keys": len(all_metric_keys),
            "status_counts": dict(sorted(status_counts.items())),
            "importance_counts": dict(sorted(importance_counts.items())),
            "n_unmapped_contexts": len(unmapped_contexts),
            "n_metric_keys_without_registry": len(metric_keys_without_registry),
        },
        "duplicate_criterion_names": duplicate_criterion_names,
        "duplicate_scenario_axis_names": duplicate_axis_names,
        "criteria": criterion_problems,
        "scenario_axes": axis_problems,
        "unmapped_contexts": tuple(sorted(unmapped_contexts)),
        "metric_keys_without_registry": tuple(sorted(metric_keys_without_registry)),
    }
