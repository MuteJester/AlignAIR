"""Metric metadata registry for benchmark reports.

The registry is deliberately conservative: it captures the current metric names,
directions, and grading thresholds used by assay reports. Scorers still emit the
same dictionaries; this module provides one metadata source for reporting,
auditing, and future schema work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .criteria import CRITERIA


_LOWER_IS_BETTER_PARTS = (
    "mae",
    "error",
    "missing",
    "overcall",
    "undercall",
    "off_by_one",
    "overlap_rate",
    "negative_span_rate",
    "false_shm",
    "outside",
    "false_positive",
    "edit_distance",
    "memory",
    "tracemalloc",
    "rss",
    "seconds_per_read",
    "milliseconds_per_read",
    "latency",
    "runtime",
    "candidate_count",
    "rerank_count",
)


@dataclass(frozen=True)
class MetricSpec:
    """Metadata for one benchmark metric key."""

    key: str
    higher_is_better: bool
    pass_threshold: float
    warn_threshold: float
    criterion_names: tuple[str, ...]
    categories: tuple[str, ...]
    statuses: tuple[str, ...]
    importance: tuple[str, ...]
    required_outputs: tuple[str, ...] = ()
    ground_truth_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def metric_higher_is_better(metric: str) -> bool:
    """Return whether larger values are better for a metric key."""

    metric_l = metric.lower()
    return not any(part in metric_l for part in _LOWER_IS_BETTER_PARTS)


def metric_quality(metric: str, value: float) -> float:
    """Normalize a metric value to an approximate higher-is-better quality score."""

    if metric_higher_is_better(metric):
        return value
    metric_l = metric.lower()
    if metric_l.endswith("_rate") or metric_l in {"cigar_edit_distance"}:
        return 1.0 - value
    return 1.0 / (1.0 + max(value, 0.0))


def metric_thresholds(metric: str) -> dict[str, Any]:
    """Return original-unit pass/warn thresholds for a metric key."""

    metric_l = metric.lower()
    higher = metric_higher_is_better(metric_l)
    if higher:
        if metric_l == "optional_field_presence":
            return {"higher_is_better": True, "pass": 0.75, "warn": 0.25}
        if metric_l in {"required_field_presence", "parseable_airr_rate", "coordinate_parse_rate"}:
            return {"higher_is_better": True, "pass": 1.0, "warn": 0.95}
        if metric_l == "reads_per_second":
            return {"higher_is_better": True, "pass": 1.0, "warn": 0.1}
        if metric_l.endswith("_within10"):
            return {"higher_is_better": True, "pass": 0.99, "warn": 0.95}
        return {"higher_is_better": True, "pass": 0.99, "warn": 0.95}

    if metric_l == "cigar_edit_distance":
        return {"higher_is_better": False, "pass": 0.0, "warn": 2.0}
    if metric_l in {"seconds_per_read", "latency_seconds", "runtime_seconds"}:
        return {"higher_is_better": False, "pass": 1.0, "warn": 10.0}
    if metric_l in {"milliseconds_per_read", "latency_ms", "runtime_ms"}:
        return {"higher_is_better": False, "pass": 1000.0, "warn": 10000.0}
    if metric_l in {"candidate_count", "rerank_count"}:
        return {"higher_is_better": False, "pass": 100.0, "warn": 1000.0}
    if metric_l.endswith("_rate") or "rate" in metric_l:
        return {"higher_is_better": False, "pass": 0.01, "warn": 0.05}
    if metric_l.endswith("_mae"):
        if "mutation_rate" in metric_l or "identity" in metric_l:
            return {"higher_is_better": False, "pass": 0.01, "warn": 0.03}
        return {"higher_is_better": False, "pass": 0.5, "warn": 2.0}
    if "memory" in metric_l or "tracemalloc" in metric_l or "rss" in metric_l:
        return {"higher_is_better": False, "pass": 4096.0, "warn": 16384.0}
    return {"higher_is_better": False, "pass": 0.0, "warn": 1.0}


def grade_metric_value(metric: str, value: float) -> dict[str, Any]:
    """Grade one metric value using registry thresholds."""

    thresholds = metric_thresholds(metric)
    higher = bool(thresholds["higher_is_better"])
    if higher:
        grade = "pass" if value >= thresholds["pass"] else "warn" if value >= thresholds["warn"] else "fail"
    else:
        grade = "pass" if value <= thresholds["pass"] else "warn" if value <= thresholds["warn"] else "fail"
    return {
        "grade": grade,
        "higher_is_better": higher,
        "pass_threshold": thresholds["pass"],
        "warn_threshold": thresholds["warn"],
        "quality_score": metric_quality(metric, value),
    }


def _build_metric_registry() -> dict[str, MetricSpec]:
    grouped: dict[str, dict[str, set[str]]] = {}
    for criterion in CRITERIA:
        for key in criterion.metric_keys:
            row = grouped.setdefault(
                key,
                {
                    "criterion_names": set(),
                    "categories": set(),
                    "statuses": set(),
                    "importance": set(),
                    "required_outputs": set(),
                    "ground_truth_fields": set(),
                },
            )
            row["criterion_names"].add(criterion.name)
            row["categories"].add(criterion.category)
            row["statuses"].add(criterion.status)
            row["importance"].add(criterion.importance)
            row["required_outputs"].update(criterion.required_outputs)
            row["ground_truth_fields"].update(criterion.ground_truth_fields)

    registry = {}
    for key, row in grouped.items():
        thresholds = metric_thresholds(key)
        registry[key] = MetricSpec(
            key=key,
            higher_is_better=bool(thresholds["higher_is_better"]),
            pass_threshold=float(thresholds["pass"]),
            warn_threshold=float(thresholds["warn"]),
            criterion_names=tuple(sorted(row["criterion_names"])),
            categories=tuple(sorted(row["categories"])),
            statuses=tuple(sorted(row["statuses"])),
            importance=tuple(sorted(row["importance"])),
            required_outputs=tuple(sorted(row["required_outputs"])),
            ground_truth_fields=tuple(sorted(row["ground_truth_fields"])),
        )
    return dict(sorted(registry.items()))


METRIC_REGISTRY: dict[str, MetricSpec] = _build_metric_registry()


def metric_registry() -> dict[str, MetricSpec]:
    """Return a copy of the current metric registry."""

    return dict(METRIC_REGISTRY)


def metric_spec(metric: str) -> MetricSpec:
    """Return metadata for one metric key.

    Unknown metrics receive direction and thresholds from the default registry
    rules. This keeps reports robust while audit still flags orphan metrics.
    """

    spec = METRIC_REGISTRY.get(metric)
    if spec is not None:
        return spec
    thresholds = metric_thresholds(metric)
    return MetricSpec(
        key=metric,
        higher_is_better=bool(thresholds["higher_is_better"]),
        pass_threshold=float(thresholds["pass"]),
        warn_threshold=float(thresholds["warn"]),
        criterion_names=(),
        categories=(),
        statuses=(),
        importance=(),
    )


def metric_spec_catalog() -> list[dict[str, Any]]:
    """Return all catalog metric specs as serializable dictionaries."""

    return [spec.to_dict() for spec in METRIC_REGISTRY.values()]
