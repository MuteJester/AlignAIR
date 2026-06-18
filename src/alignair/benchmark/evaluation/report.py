"""Assay-style reporting for benchmark score dictionaries."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..core import GENES, criteria_catalog

_LOWER_IS_BETTER_PARTS = (
    "mae",
    "error",
    "missing",
    "overcall",
    "undercall",
    "outside",
    "false_positive",
    "edit_distance",
    "memory",
    "candidate_count",
    "rerank_count",
)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _higher_is_better(metric: str) -> bool:
    return not any(part in metric for part in _LOWER_IS_BETTER_PARTS)


def _quality(metric: str, value: float) -> float:
    if _higher_is_better(metric):
        return value
    if metric.endswith("_rate") or metric in {"cigar_edit_distance"}:
        return 1.0 - value
    return 1.0 / (1.0 + max(value, 0.0))


def _extract_score_root(scores_or_report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "results" in scores_or_report:
        results = scores_or_report.get("results", {})
        return (
            results.get("overall", {}),
            results.get("by_context", {}),
            scores_or_report,
        )
    return scores_or_report, scores_or_report.get("by_stratum", {}), {}


def _metric_values_for_key(overall: dict[str, Any], key: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    global_value = _finite(overall.get("global", {}).get(key))
    if global_value is not None:
        values["global"] = global_value
    genes = overall.get("genes", {})
    gene_values = {}
    for gene in GENES:
        value = _finite(genes.get(gene, {}).get(key))
        if value is not None:
            gene_values[gene] = value
    if gene_values:
        values["genes"] = gene_values
    return values


def _iter_metric_values(scored: dict[str, Any]):
    for key, value in scored.get("global", {}).items():
        f = _finite(value)
        if f is not None:
            yield "global", None, key, f
    for gene, vals in scored.get("genes", {}).items():
        for key, value in vals.items():
            f = _finite(value)
            if f is not None:
                yield "gene", gene, key, f


def _criterion_entry(criterion: dict[str, Any], overall: dict[str, Any]) -> dict[str, Any]:
    observed = {}
    quality_values = []
    for key in criterion.get("metric_keys", ()):
        values = _metric_values_for_key(overall, key)
        if not values:
            continue
        observed[key] = values
        if "global" in values:
            quality_values.append(_quality(key, values["global"]))
        for value in values.get("genes", {}).values():
            quality_values.append(_quality(key, value))

    metric_keys = tuple(criterion.get("metric_keys", ()))
    missing = tuple(key for key in metric_keys if key not in observed)
    return {
        "name": criterion["name"],
        "category": criterion["category"],
        "status": criterion.get("status"),
        "importance": criterion.get("importance"),
        "description": criterion.get("description"),
        "interpretation": criterion.get("interpretation"),
        "metric_keys": metric_keys,
        "observed_metrics": observed,
        "missing_metric_keys": missing,
        "n_observed_metric_keys": len(observed),
        "n_metric_keys": len(metric_keys),
        "coverage_fraction": len(observed) / len(metric_keys) if metric_keys else 0.0,
        "quality_score": sum(quality_values) / len(quality_values) if quality_values else None,
    }


def _weak_contexts(
    by_context: dict[str, Any],
    metric_to_criterion: dict[str, dict[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = []
    for context, scored in by_context.items():
        for scope, gene, metric, value in _iter_metric_values(scored):
            criterion = metric_to_criterion.get(metric)
            if criterion is None:
                continue
            quality = _quality(metric, value)
            rows.append(
                {
                    "context": context,
                    "scope": scope,
                    "gene": gene,
                    "metric": metric,
                    "value": value,
                    "higher_is_better": _higher_is_better(metric),
                    "quality_score": quality,
                    "criterion": criterion["name"],
                    "category": criterion["category"],
                }
            )
    rows.sort(key=lambda row: (row["quality_score"], row["category"], row["criterion"], row["metric"]))
    return rows[:top_n]


def build_assay_report(
    scores_or_report: dict[str, Any],
    *,
    criteria: list[dict[str, Any]] | None = None,
    top_n_contexts: int = 25,
) -> dict[str, Any]:
    """Build a criterion/category-oriented report from benchmark scores.

    Accepts either raw ``score_cases`` output or a ``run_online_benchmark`` report.
    """

    overall, by_context, source_report = _extract_score_root(scores_or_report)
    criteria = criteria or source_report.get("criteria") or criteria_catalog()
    entries = [_criterion_entry(c, overall) for c in criteria]
    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["category"]].append(entry)
    for category, category_entries in sorted(grouped.items()):
        scores = [e["quality_score"] for e in category_entries if e["quality_score"] is not None]
        by_category[category] = {
            "n_criteria": len(category_entries),
            "n_with_results": sum(1 for e in category_entries if e["n_observed_metric_keys"] > 0),
            "mean_quality_score": sum(scores) / len(scores) if scores else None,
            "criteria": category_entries,
        }

    metric_to_criterion = {}
    for criterion in criteria:
        for key in criterion.get("metric_keys", ()):
            metric_to_criterion.setdefault(key, criterion)
    with_results = [e for e in entries if e["n_observed_metric_keys"] > 0]
    scores = [e["quality_score"] for e in with_results if e["quality_score"] is not None]
    return {
        "summary": {
            "n_cases": overall.get("n_cases"),
            "frame": overall.get("frame") or source_report.get("frame"),
            "n_criteria": len(entries),
            "n_criteria_with_results": len(with_results),
            "n_criteria_without_results": len(entries) - len(with_results),
            "mean_quality_score": sum(scores) / len(scores) if scores else None,
        },
        "by_category": by_category,
        "criteria": entries,
        "weak_contexts": _weak_contexts(by_context, metric_to_criterion, top_n=top_n_contexts),
    }
