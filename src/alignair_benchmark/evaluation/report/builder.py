from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ...core import GENES, criteria_catalog

from .grading import (
    _finite,
    _quality,
    _worst_grade,
    _higher_is_better,
    _metric_assessments,
    _criterion_grade,
)
from .completeness import _apply_completeness_gate


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
    coverage_fraction = len(observed) / len(metric_keys) if metric_keys else 0.0
    assessments = _metric_assessments(observed)
    grade, reasons = _criterion_grade(
        status=criterion.get("status"),
        coverage_fraction=coverage_fraction,
        missing_metric_keys=missing,
        metric_assessments=assessments,
    )
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
        "coverage_fraction": coverage_fraction,
        "quality_score": sum(quality_values) / len(quality_values) if quality_values else None,
        "grade": grade,
        "grade_reasons": reasons,
        "metric_assessments": assessments,
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
    completeness_gate = _apply_completeness_gate(entries, source_report)
    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["category"]].append(entry)
    for category, category_entries in sorted(grouped.items()):
        scores = [e["quality_score"] for e in category_entries if e["quality_score"] is not None]
        category_grades = [
            e["grade"]
            for e in category_entries
            if e["grade"] not in {"not_scored", "planned"}
        ]
        grade_counts = Counter(e["grade"] for e in category_entries)
        by_category[category] = {
            "n_criteria": len(category_entries),
            "n_with_results": sum(1 for e in category_entries if e["n_observed_metric_keys"] > 0),
            "mean_quality_score": sum(scores) / len(scores) if scores else None,
            "grade": _worst_grade(category_grades),
            "grade_counts": dict(sorted(grade_counts.items())),
            "criteria": category_entries,
        }

    metric_to_criterion = {}
    for criterion in criteria:
        for key in criterion.get("metric_keys", ()):
            metric_to_criterion.setdefault(key, criterion)
    with_results = [e for e in entries if e["n_observed_metric_keys"] > 0]
    scores = [e["quality_score"] for e in with_results if e["quality_score"] is not None]
    scored_entries = [e for e in entries if e["grade"] not in {"not_scored", "planned"}]
    grade_counts = Counter(e["grade"] for e in entries)
    failed = [e for e in scored_entries if e["grade"] == "fail"]
    warned = [e for e in scored_entries if e["grade"] == "warn"]
    return {
        "summary": {
            "n_cases": overall.get("n_cases"),
            "frame": overall.get("frame") or source_report.get("frame"),
            "n_criteria": len(entries),
            "n_criteria_with_results": len(with_results),
            "n_criteria_without_results": len(entries) - len(with_results),
            "mean_quality_score": sum(scores) / len(scores) if scores else None,
            "grade": _worst_grade([e["grade"] for e in scored_entries]),
            "grade_counts": dict(sorted(grade_counts.items())),
            "n_failed_criteria": len(failed),
            "n_warned_criteria": len(warned),
            "completeness_gate_grade": completeness_gate["grade"],
            "n_blocking_unscored_core_criteria": len(completeness_gate["blocking_unscored_core_criteria"]),
            "n_truth_unavailable_core_criteria": len(completeness_gate["truth_unavailable_core_criteria"]),
        },
        "completeness_gate": completeness_gate,
        "by_category": by_category,
        "criteria": entries,
        "critical_failures": [
            {
                "name": e["name"],
                "category": e["category"],
                "importance": e.get("importance"),
                "quality_score": e.get("quality_score"),
                "grade_reasons": e.get("grade_reasons", ()),
            }
            for e in failed
            if e.get("importance") == "core"
        ],
        "weak_contexts": _weak_contexts(by_context, metric_to_criterion, top_n=top_n_contexts),
    }
