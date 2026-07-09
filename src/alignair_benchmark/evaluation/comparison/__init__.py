"""Paired model-vs-model benchmark comparison reports."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

from ...core.artifacts import MODEL_COMPARISON_REPORT, artifact_metadata
from ...core.schema import BenchmarkCase
from .decision import (
    build_decision_report as _build_decision_report,
    corrected_confidence as _corrected_confidence,
    decision_family_size as _decision_family_size,
    merge_metric_lists as _merge_metric_lists,
)
from .policy import (
    COMPARISON_POLICY_TEMPLATES,
    MULTIPLE_COMPARISON_CORRECTIONS,
    comparison_policy_catalog,
    policy_template as _policy_template,
    validate_comparison_policy_catalog,
)
from .scoring import compare_scope as _compare_scope
from ..matching import align_predictions_to_cases
from ..uncertainty import DEFAULT_BOOTSTRAP_METRICS

DEFAULT_COMPARISON_METRICS: tuple[str, ...] = DEFAULT_BOOTSTRAP_METRICS


def _merged_metric_paths(
    metric_paths: Iterable[str] | None,
    primary_metrics: tuple[str, ...],
    guardrail_metrics: tuple[str, ...],
) -> tuple[str, ...]:
    out = []
    seen = set()
    for source in (tuple(metric_paths or DEFAULT_COMPARISON_METRICS), primary_metrics, guardrail_metrics):
        for path in source:
            if path not in seen:
                out.append(path)
                seen.add(path)
    return tuple(out)


def build_model_comparison_report(
    cases: Iterable[BenchmarkCase],
    predictions_a: Iterable[dict[str, Any] | None],
    predictions_b: Iterable[dict[str, Any] | None],
    *,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    frame: str = "canonical",
    metric_paths: Iterable[str] | None = None,
    match_by: str | None = None,
    duplicate_policy: str = "first",
    n_bootstrap: int = 0,
    confidence: float = 0.95,
    seed: int = 123,
    include_strata: bool = True,
    min_stratum_cases: int = 2,
    practical_delta: float = 0.0,
    case_tie_tolerance: float = 0.0,
    metric_directions: dict[str, str] | None = None,
    comparison_policy: str | None = None,
    multiple_comparison_correction: str = "none",
    primary_metrics: Iterable[str] | None = None,
    guardrail_metrics: Iterable[str] | None = None,
    minimum_primary_advantage: float | None = None,
    maximum_guardrail_regression: float | None = None,
    include_expensive_record_fields: bool = True,
) -> dict[str, Any]:
    """Compare two prediction sets on the same GenAIRR benchmark cases.

    Raw deltas are reported as ``model_b - model_a``. ``model_b_advantage`` is
    direction-adjusted, so positive values always favor model B.
    """

    case_list = list(cases)
    preds_a = list(predictions_a)
    preds_b = list(predictions_b)
    match_report: dict[str, Any] = {}
    if match_by and match_by != "order":
        match_a = align_predictions_to_cases(case_list, preds_a, id_field=match_by, duplicate_policy=duplicate_policy)
        match_b = align_predictions_to_cases(case_list, preds_b, id_field=match_by, duplicate_policy=duplicate_policy)
        preds_a = match_a.predictions
        preds_b = match_b.predictions
        match_report = {
            "model_a": match_a.report,
            "model_b": match_b.report,
        }
    else:
        if len(case_list) != len(preds_a):
            raise ValueError(f"case/model A prediction length mismatch: {len(case_list)} != {len(preds_a)}")
        if len(case_list) != len(preds_b):
            raise ValueError(f"case/model B prediction length mismatch: {len(case_list)} != {len(preds_b)}")

    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if n_bootstrap and not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if practical_delta < 0:
        raise ValueError("practical_delta must be non-negative")
    if case_tie_tolerance < 0:
        raise ValueError("case_tie_tolerance must be non-negative")
    policy = _policy_template(comparison_policy)
    policy_primary = tuple(policy.get("primary_metrics", ())) if policy else ()
    policy_guardrails = tuple(policy.get("guardrail_metrics", ())) if policy else ()
    resolved_minimum_primary_advantage = (
        float(policy.get("minimum_primary_advantage", 0.0))
        if minimum_primary_advantage is None and policy
        else float(minimum_primary_advantage or 0.0)
    )
    resolved_maximum_guardrail_regression = (
        float(policy.get("maximum_guardrail_regression", 0.0))
        if maximum_guardrail_regression is None and policy
        else float(maximum_guardrail_regression or 0.0)
    )
    if resolved_minimum_primary_advantage < 0:
        raise ValueError("minimum_primary_advantage must be non-negative")
    if resolved_maximum_guardrail_regression < 0:
        raise ValueError("maximum_guardrail_regression must be non-negative")

    primary_paths = _merge_metric_lists(policy_primary, primary_metrics)
    guardrail_paths = _merge_metric_lists(policy_guardrails, guardrail_metrics)
    family_size = _decision_family_size(primary_paths, guardrail_paths)
    corrected_decision_confidence = None
    if n_bootstrap and family_size:
        corrected_decision_confidence = _corrected_confidence(
            confidence,
            family_size,
            multiple_comparison_correction,
        )
    else:
        _corrected_confidence(confidence, max(family_size, 1), multiple_comparison_correction)
    multiple_comparison = {
        "method": multiple_comparison_correction,
        "family_size": family_size,
        "family_confidence": confidence if n_bootstrap and family_size else None,
        "per_metric_confidence": corrected_decision_confidence,
        "applied": bool(
            n_bootstrap
            and family_size > 1
            and multiple_comparison_correction != "none"
        ),
    }
    paths = _merged_metric_paths(metric_paths, primary_paths, guardrail_paths)
    rng = random.Random(seed)
    overall = _compare_scope(
        case_list,
        preds_a,
        preds_b,
        frame=frame,
        metric_paths=paths,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        decision_confidence=corrected_decision_confidence,
        decision_family_size=family_size,
        multiple_comparison_correction=multiple_comparison_correction,
        practical_delta=practical_delta,
        case_tie_tolerance=case_tie_tolerance,
        metric_directions=metric_directions,
        include_expensive_record_fields=include_expensive_record_fields,
        rng=rng,
    )
    report: dict[str, Any] = {
        "artifact": artifact_metadata(MODEL_COMPARISON_REPORT),
        "comparison": {
            "method": "paired_case_delta",
            "truth_source": "GenAIRR benchmark cases",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "n_cases": len(case_list),
            "frame": frame,
            "metric_paths": list(paths),
            "raw_delta": "model_b - model_a",
            "model_b_advantage": "direction-adjusted delta; positive favors model_b",
            "n_bootstrap": n_bootstrap,
            "confidence": confidence if n_bootstrap else None,
            "seed": seed if n_bootstrap else None,
            "practical_delta": practical_delta,
            "case_tie_tolerance": case_tie_tolerance,
            "comparison_policy": comparison_policy,
            "multiple_comparison_correction": multiple_comparison_correction,
            "primary_metrics": list(primary_paths),
            "guardrail_metrics": list(guardrail_paths),
            "minimum_primary_advantage": resolved_minimum_primary_advantage,
            "maximum_guardrail_regression": resolved_maximum_guardrail_regression,
        },
        "summary": overall["summary"],
        "overall": overall["metrics"],
        "by_stratum": {},
        "skipped_strata": [],
    }
    if match_report:
        report["prediction_matching"] = match_report
    decision = _build_decision_report(
        overall["metrics"],
        policy_name=comparison_policy,
        policy_description=policy.get("description") if policy else None,
        primary_metrics=primary_paths,
        guardrail_metrics=guardrail_paths,
        minimum_primary_advantage=resolved_minimum_primary_advantage,
        maximum_guardrail_regression=resolved_maximum_guardrail_regression,
        multiple_comparison=multiple_comparison,
    )
    if decision is not None:
        report["decision"] = decision

    if include_strata:
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for idx, case in enumerate(case_list):
            by_stratum[case.stratum].append(idx)
        for stratum, indices in sorted(by_stratum.items()):
            if n_bootstrap and len(indices) < min_stratum_cases:
                report["skipped_strata"].append(
                    {
                        "stratum": stratum,
                        "n_cases": len(indices),
                        "reason": f"requires at least {min_stratum_cases} cases",
                    }
                )
                continue
            scope = _compare_scope(
                [case_list[i] for i in indices],
                [preds_a[i] for i in indices],
                [preds_b[i] for i in indices],
                frame=frame,
                metric_paths=paths,
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                decision_confidence=corrected_decision_confidence,
                decision_family_size=family_size,
                multiple_comparison_correction=multiple_comparison_correction,
                practical_delta=practical_delta,
                case_tie_tolerance=case_tie_tolerance,
                metric_directions=metric_directions,
                include_expensive_record_fields=include_expensive_record_fields,
                rng=rng,
            )
            report["by_stratum"][stratum] = scope
    return report
