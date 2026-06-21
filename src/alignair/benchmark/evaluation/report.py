"""Assay-style reporting for benchmark score dictionaries."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from ..core import GENES, criteria_catalog

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
    "candidate_count",
    "rerank_count",
)

_GRADE_RANK = {"pass": 0, "warn": 1, "fail": 2, "not_scored": 3, "planned": 4}


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


def _thresholds_for_metric(metric: str) -> dict[str, Any]:
    """Return original-unit pass/warn thresholds for a metric."""

    metric_l = metric.lower()
    higher = _higher_is_better(metric_l)
    if higher:
        if metric_l == "optional_field_presence":
            return {"higher_is_better": True, "pass": 0.75, "warn": 0.25}
        if metric_l in {"required_field_presence", "parseable_airr_rate", "coordinate_parse_rate"}:
            return {"higher_is_better": True, "pass": 1.0, "warn": 0.95}
        if metric_l.endswith("_within10"):
            return {"higher_is_better": True, "pass": 0.99, "warn": 0.95}
        return {"higher_is_better": True, "pass": 0.99, "warn": 0.95}

    if metric_l == "cigar_edit_distance":
        return {"higher_is_better": False, "pass": 0.0, "warn": 2.0}
    if metric_l.endswith("_rate") or "rate" in metric_l:
        return {"higher_is_better": False, "pass": 0.01, "warn": 0.05}
    if metric_l.endswith("_mae"):
        if "mutation_rate" in metric_l or "identity" in metric_l:
            return {"higher_is_better": False, "pass": 0.01, "warn": 0.03}
        return {"higher_is_better": False, "pass": 0.5, "warn": 2.0}
    if "memory" in metric_l:
        return {"higher_is_better": False, "pass": 4096.0, "warn": 16384.0}
    return {"higher_is_better": False, "pass": 0.0, "warn": 1.0}


def _grade_metric(metric: str, value: float) -> dict[str, Any]:
    thresholds = _thresholds_for_metric(metric)
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
        "quality_score": _quality(metric, value),
    }


def _worst_grade(grades: list[str]) -> str:
    if not grades:
        return "not_scored"
    return max(grades, key=lambda grade: _GRADE_RANK.get(grade, -1))


def _metric_assessments(observed: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, values in observed.items():
        global_value = _finite(values.get("global"))
        if global_value is not None:
            rows.append(
                {
                    "metric": metric,
                    "scope": "global",
                    "gene": None,
                    "value": global_value,
                    **_grade_metric(metric, global_value),
                }
            )
        for gene, value in values.get("genes", {}).items():
            f = _finite(value)
            if f is None:
                continue
            rows.append(
                {
                    "metric": metric,
                    "scope": "gene",
                    "gene": gene,
                    "value": f,
                    **_grade_metric(metric, f),
                }
            )
    return rows


def _criterion_grade(
    *,
    status: str | None,
    coverage_fraction: float,
    missing_metric_keys: tuple[str, ...],
    metric_assessments: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    if not metric_assessments:
        if status == "planned":
            return "planned", ["criterion is planned and has no implemented metrics yet"]
        return "not_scored", ["no metric values were observed for this criterion"]

    grades = [row["grade"] for row in metric_assessments]
    grade = _worst_grade(grades)
    reasons: list[str] = []
    counts = Counter(grades)
    if counts.get("fail"):
        reasons.append(f"{counts['fail']} metric value(s) below fail threshold")
    if counts.get("warn"):
        reasons.append(f"{counts['warn']} metric value(s) in warning range")
    if missing_metric_keys and status == "available":
        if coverage_fraction < 0.5:
            grade = "fail"
            reasons.append("less than half of expected metric keys are present")
        elif grade == "pass":
            grade = "warn"
            reasons.append("some expected metric keys are missing")
    if not reasons:
        reasons.append("all observed metric values passed configured thresholds")
    return grade, reasons


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


def _apply_completeness_gate(
    entries: list[dict[str, Any]],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    """Fail full reports when available core criteria are unscored."""

    audit = source_report.get("criteria_audit") or {}
    audit_summary = audit.get("summary") or {}
    has_truth_audit = bool(audit_summary.get("has_case_truth_audit"))
    if not has_truth_audit:
        return {
            "applied": False,
            "grade": "pass",
            "reason": "case-level truth audit unavailable; completeness gate was not applied",
            "blocking_unscored_core_criteria": [],
            "truth_unavailable_core_criteria": [],
            "partial_unscored_core_criteria": [],
        }

    audit_by_name = {
        row.get("name"): row
        for row in audit.get("criteria", ())
        if row.get("name")
    }
    blocking = []
    truth_unavailable = []
    partial_unscored = []
    for entry in entries:
        if entry.get("importance") != "core" or entry.get("n_observed_metric_keys", 0) > 0:
            continue
        audit_row = audit_by_name.get(entry["name"], {})
        unavailable_truth = tuple(audit_row.get("unavailable_truth_fields", ()))
        row = {
            "name": entry["name"],
            "category": entry["category"],
            "status": entry.get("status"),
            "missing_metric_keys": entry.get("missing_metric_keys", ()),
            "unavailable_truth_fields": unavailable_truth,
        }
        if entry.get("status") == "available":
            if unavailable_truth:
                truth_unavailable.append(row)
            else:
                blocking.append(row)
                entry["grade"] = "fail"
                entry["grade_reasons"] = list(entry.get("grade_reasons", ())) + [
                    "available core criterion has no observed metrics despite available truth fields",
                ]
                entry["completeness_gate_failed"] = True
        elif entry.get("status") == "partial":
            partial_unscored.append(row)

    return {
        "applied": True,
        "grade": "fail" if blocking else "pass",
        "reason": (
            "available core criteria without observed metrics block a pass"
            if blocking
            else "no available core criteria were missing scoreable metrics"
        ),
        "blocking_unscored_core_criteria": blocking,
        "truth_unavailable_core_criteria": truth_unavailable,
        "partial_unscored_core_criteria": partial_unscored,
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
