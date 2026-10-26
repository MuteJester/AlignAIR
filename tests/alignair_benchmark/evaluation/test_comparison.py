"""Paired model comparisons and explicit endpoint policies."""
import pytest
from alignair_benchmark import (
    case_to_prediction,
    build_model_comparison_report,
    comparison_policy_catalog,
)
from tests.alignair_benchmark._support import _manual_case

pytestmark = pytest.mark.component
pytest.importorskip("GenAIRR")


def test_model_comparison_reports_paired_deltas_and_verdicts():
    cases = [
        _manual_case("cmp1", ("IGHV1-1*01",)),
        _manual_case("cmp2", ("IGHV1-2*01",)),
    ]
    preds_a = []
    preds_b = []
    for case in cases:
        pred_a = case_to_prediction(case)
        pred_a["v_call"] = "IGHV9-9*01"
        pred_a["v_calls"] = ["IGHV9-9*01"]
        pred_b = case_to_prediction(case)
        pred_b["v_sequence_start"] = case.genes["v"].sequence_start + 10
        preds_a.append(pred_a)
        preds_b.append(pred_b)

    report = build_model_comparison_report(
        cases,
        preds_a,
        preds_b,
        model_a_name="old",
        model_b_name="new",
        metric_paths=("genes.v.call_top1_in_set",),
        primary_metrics=("genes.v.call_top1_in_set",),
        guardrail_metrics=("genes.v.ss_mae",),
        n_bootstrap=10,
        confidence=0.9,
        seed=5,
        include_strata=False,
    )

    call = report["overall"]["genes.v.call_top1_in_set"]
    assert call["metric_key"] == "call_top1_in_set"
    assert call["direction"] == "higher_is_better"
    assert call["direction_source"] == "metric_registry"
    assert call["metric_spec"]["registered"] is True
    assert call["metric_spec"]["key"] == "call_top1_in_set"
    assert call["metric_spec"]["higher_is_better"] is True
    assert call["model_a"] == 0.0
    assert call["model_b"] == 1.0
    assert call["raw_delta_model_b_minus_model_a"] == 1.0
    assert call["model_b_advantage"] == 1.0
    assert call["verdict"] == "model_b_better"
    assert call["preferred_model"] == "new"
    assert call["win_loss_tie"]["model_b_wins"] == 2
    assert call["model_b_advantage_ci_low"] == 1.0

    mae = report["overall"]["genes.v.ss_mae"]
    assert mae["direction"] == "lower_is_better"
    assert mae["direction_source"] == "metric_registry"
    assert mae["metric_spec"]["key"] == "ss_mae"
    assert mae["metric_spec"]["higher_is_better"] is False
    assert mae["model_a"] == 0.0
    assert mae["model_b"] == 10.0
    assert mae["raw_delta_model_b_minus_model_a"] == 10.0
    assert mae["model_b_advantage"] == -10.0
    assert mae["verdict"] == "model_a_better"
    assert mae["preferred_model"] == "old"
    assert report["summary"]["verdict"] == "mixed"

    decision = report["decision"]
    assert decision["verdict"] == "blocked_by_guardrail_regression"
    assert decision["primary_endpoints"][0]["status"] == "pass"
    assert decision["guardrails"][0]["status"] == "fail"
    assert decision["guardrails"][0]["metric"] == "genes.v.ss_mae"


def test_model_comparison_metric_direction_override_is_explicit():
    cases = [
        _manual_case("cmp_override1", ("IGHV1-1*01",)),
        _manual_case("cmp_override2", ("IGHV1-2*01",)),
    ]
    preds_a = []
    preds_b = []
    for case in cases:
        pred_a = case_to_prediction(case)
        pred_a["v_call"] = "IGHV9-9*01"
        pred_a["v_calls"] = ["IGHV9-9*01"]
        preds_a.append(pred_a)
        preds_b.append(case_to_prediction(case))

    report = build_model_comparison_report(
        cases,
        preds_a,
        preds_b,
        model_a_name="old",
        model_b_name="new",
        metric_paths=("genes.v.call_top1_in_set",),
        metric_directions={"genes.v.call_top1_in_set": "lower"},
        include_strata=False,
    )

    row = report["overall"]["genes.v.call_top1_in_set"]
    assert row["direction"] == "lower_is_better"
    assert row["direction_source"] == "override"
    assert row["metric_spec"]["higher_is_better"] is True
    assert row["raw_delta_model_b_minus_model_a"] == 1.0
    assert row["model_b_advantage"] == -1.0
    assert row["verdict"] == "model_a_better"
    assert row["preferred_model"] == "old"


def test_named_comparison_policy_populates_endpoint_gates():
    cases = [
        _manual_case("policy_cmp1", ("IGHV1-1*01",)),
        _manual_case("policy_cmp2", ("IGHV1-2*01",)),
    ]
    for case in cases:
        case.record["junction"] = "AACCGG"
        case.scalars["productive"] = 1.0
    preds_a = []
    preds_b = []
    for case in cases:
        pred_a = case_to_prediction(case)
        pred_a["v_call"] = "IGHV9-9*01"
        pred_a["v_calls"] = ["IGHV9-9*01"]
        pred_a["j_call"] = "IGHJ9*99"
        pred_a["j_calls"] = ["IGHJ9*99"]
        preds_a.append(pred_a)
        preds_b.append(case_to_prediction(case))

    catalog = {row["name"]: row for row in comparison_policy_catalog()}
    report = build_model_comparison_report(
        cases,
        preds_a,
        preds_b,
        model_a_name="old",
        model_b_name="new",
        comparison_policy="allele_calling_core",
        multiple_comparison_correction="sidak",
        n_bootstrap=10,
        confidence=0.9,
        seed=7,
        include_strata=False,
    )

    assert "allele_calling_core" in catalog
    assert report["comparison"]["comparison_policy"] == "allele_calling_core"
    assert report["decision"]["policy"] == "allele_calling_core"
    assert report["decision"]["primary_metrics"] == catalog["allele_calling_core"]["primary_metrics"]
    correction = report["decision"]["multiple_comparison"]
    family_size = len(set(report["decision"]["primary_metrics"] + report["decision"]["guardrail_metrics"]))
    assert correction["method"] == "sidak"
    assert correction["applied"] is True
    assert correction["family_size"] == family_size
    assert correction["family_confidence"] == 0.9
    assert correction["per_metric_confidence"] == pytest.approx(0.9 ** (1 / family_size))
    assert report["decision"]["primary_endpoints"][0]["basis"] == "multiple_comparison_adjusted_bootstrap_ci"
    assert report["decision"]["verdict"] == "model_b_superior"
    assert report["decision"]["status_counts"]["pass"] == (
        len(report["decision"]["primary_endpoints"]) + len(report["decision"]["guardrails"])
    )
