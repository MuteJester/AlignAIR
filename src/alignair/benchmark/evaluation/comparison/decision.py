"""Endpoint and guardrail decision gates for paired comparison reports."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .policy import MULTIPLE_COMPARISON_CORRECTIONS


def merge_metric_lists(*sources: Iterable[str] | None) -> tuple[str, ...]:
    out = []
    seen = set()
    for source in sources:
        for path in source or ():
            if path not in seen:
                out.append(path)
                seen.add(path)
    return tuple(out)


def decision_family_size(primary_metrics: tuple[str, ...], guardrail_metrics: tuple[str, ...]) -> int:
    return len(merge_metric_lists(primary_metrics, guardrail_metrics))


def corrected_confidence(confidence: float, family_size: int, method: str) -> float:
    if method not in MULTIPLE_COMPARISON_CORRECTIONS:
        choices = ", ".join(MULTIPLE_COMPARISON_CORRECTIONS)
        raise ValueError(f"multiple_comparison_correction must be one of: {choices}")
    if family_size <= 1 or method == "none":
        return confidence
    alpha = 1.0 - confidence
    if method == "bonferroni":
        return 1.0 - (alpha / family_size)
    if method == "sidak":
        return confidence ** (1.0 / family_size)
    return confidence


def _comparison_interval(row: dict[str, Any]) -> tuple[float | None, float | None, str]:
    decision_low = row.get("decision_model_b_advantage_ci_low")
    decision_high = row.get("decision_model_b_advantage_ci_high")
    if decision_low is not None and decision_high is not None:
        return decision_low, decision_high, "multiple_comparison_adjusted_bootstrap_ci"
    low = row.get("model_b_advantage_ci_low")
    high = row.get("model_b_advantage_ci_high")
    if low is not None and high is not None:
        return low, high, "bootstrap_ci"
    return None, None, "point_estimate"


def _primary_gate_row(metric: str, row: dict[str, Any] | None, threshold: float) -> dict[str, Any]:
    advantage = row.get("model_b_advantage") if row else None
    ci_low, ci_high, basis = _comparison_interval(row or {})
    out = {
        "metric": metric,
        "role": "primary",
        "basis": basis,
        "model_b_advantage": advantage,
        "model_b_advantage_ci_low": ci_low,
        "model_b_advantage_ci_high": ci_high,
        "decision_confidence": row.get("decision_confidence") if row else None,
        "multiple_comparison_correction": row.get("decision_multiple_comparison_correction") if row else None,
        "multiple_comparison_family_size": row.get("decision_family_size") if row else None,
        "minimum_required_advantage": threshold,
        "status": "not_scored",
        "reason": "metric was not scored for both models",
    }
    if advantage is None:
        return out
    if ci_low is not None and ci_high is not None:
        if ci_low > threshold:
            out.update(status="pass", reason="confidence interval clears the primary improvement threshold")
        elif ci_high <= threshold:
            out.update(status="fail", reason="confidence interval does not clear the primary improvement threshold")
        else:
            out.update(status="inconclusive", reason="confidence interval overlaps the primary improvement threshold")
        return out
    if advantage > threshold:
        out.update(status="pass", reason="point estimate clears the primary improvement threshold")
    else:
        out.update(status="fail", reason="point estimate does not clear the primary improvement threshold")
    return out


def _guardrail_gate_row(metric: str, row: dict[str, Any] | None, max_regression: float) -> dict[str, Any]:
    advantage = row.get("model_b_advantage") if row else None
    ci_low, ci_high, basis = _comparison_interval(row or {})
    floor = -max_regression
    out = {
        "metric": metric,
        "role": "guardrail",
        "basis": basis,
        "model_b_advantage": advantage,
        "model_b_advantage_ci_low": ci_low,
        "model_b_advantage_ci_high": ci_high,
        "decision_confidence": row.get("decision_confidence") if row else None,
        "multiple_comparison_correction": row.get("decision_multiple_comparison_correction") if row else None,
        "multiple_comparison_family_size": row.get("decision_family_size") if row else None,
        "maximum_allowed_regression": max_regression,
        "minimum_allowed_advantage": floor,
        "status": "not_scored",
        "reason": "metric was not scored for both models",
    }
    if advantage is None:
        return out
    if ci_low is not None and ci_high is not None:
        if ci_low >= floor:
            out.update(status="pass", reason="confidence interval rules out unacceptable regression")
        elif ci_high < floor:
            out.update(status="fail", reason="confidence interval shows unacceptable regression")
        else:
            out.update(status="inconclusive", reason="confidence interval overlaps the regression limit")
        return out
    if advantage >= floor:
        out.update(status="pass", reason="point estimate stays within the regression limit")
    else:
        out.update(status="fail", reason="point estimate exceeds the regression limit")
    return out


def _decision_verdict(primary_rows: list[dict[str, Any]], guardrail_rows: list[dict[str, Any]]) -> str:
    rows = primary_rows + guardrail_rows
    if not rows:
        return "not_configured"
    if any(row["status"] == "not_scored" for row in rows):
        return "not_scored"
    if any(row["status"] == "fail" for row in guardrail_rows):
        return "blocked_by_guardrail_regression"
    if any(row["status"] == "inconclusive" for row in rows):
        return "inconclusive"
    if primary_rows:
        if all(row["status"] == "pass" for row in primary_rows):
            return "model_b_superior"
        return "no_primary_improvement"
    if guardrail_rows and all(row["status"] == "pass" for row in guardrail_rows):
        return "no_regression_pass"
    return "inconclusive"


def build_decision_report(
    rows: dict[str, dict[str, Any]],
    *,
    policy_name: str | None,
    policy_description: str | None,
    primary_metrics: tuple[str, ...],
    guardrail_metrics: tuple[str, ...],
    minimum_primary_advantage: float,
    maximum_guardrail_regression: float,
    multiple_comparison: dict[str, Any],
) -> dict[str, Any] | None:
    if not primary_metrics and not guardrail_metrics:
        return None
    primary_rows = [
        _primary_gate_row(metric, rows.get(metric), minimum_primary_advantage)
        for metric in primary_metrics
    ]
    guardrail_rows = [
        _guardrail_gate_row(metric, rows.get(metric), maximum_guardrail_regression)
        for metric in guardrail_metrics
    ]
    status_counts: dict[str, int] = defaultdict(int)
    for row in primary_rows + guardrail_rows:
        status_counts[row["status"]] += 1
    return {
        "policy": policy_name or "custom",
        "policy_type": "primary_endpoint_with_no_regression_guardrails",
        "policy_description": policy_description,
        "verdict": _decision_verdict(primary_rows, guardrail_rows),
        "primary_metrics": list(primary_metrics),
        "guardrail_metrics": list(guardrail_metrics),
        "minimum_primary_advantage": minimum_primary_advantage,
        "maximum_guardrail_regression": maximum_guardrail_regression,
        "multiple_comparison": multiple_comparison,
        "status_counts": dict(sorted(status_counts.items())),
        "primary_endpoints": primary_rows,
        "guardrails": guardrail_rows,
    }
