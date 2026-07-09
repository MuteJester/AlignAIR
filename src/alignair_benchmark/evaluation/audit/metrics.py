from __future__ import annotations

import math
from typing import Any

from ...core import metric_spec


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _extract_overall(scores_or_report: dict[str, Any] | None) -> dict[str, Any]:
    payload = scores_or_report or {}
    if "results" in payload:
        return payload.get("results", {}).get("overall", {})
    return payload


def _observed_metric_keys(scores_or_report: dict[str, Any] | None) -> dict[str, Any]:
    overall = _extract_overall(scores_or_report)
    global_keys = {
        key
        for key, value in overall.get("global", {}).items()
        if _finite(value) is not None
    }
    gene_keys: dict[str, set[str]] = {}
    for gene, metrics in overall.get("genes", {}).items():
        gene_keys[gene] = {
            key for key, value in metrics.items() if _finite(value) is not None
        }
    combined = set(global_keys)
    for keys in gene_keys.values():
        combined.update(keys)
    return {
        "global": sorted(global_keys),
        "genes": {gene: sorted(keys) for gene, keys in sorted(gene_keys.items())},
        "all": sorted(combined),
    }


def _audit_metric_spec(metric: str) -> dict[str, Any]:
    """Return stable, audit-facing registry metadata for a metric key."""

    spec = metric_spec(metric)
    return {
        "key": spec.key,
        "registered": bool(spec.criterion_names),
        "higher_is_better": spec.higher_is_better,
        "pass_threshold": spec.pass_threshold,
        "warn_threshold": spec.warn_threshold,
        "criteria": spec.criterion_names,
        "categories": spec.categories,
        "statuses": spec.statuses,
        "importance": spec.importance,
    }
