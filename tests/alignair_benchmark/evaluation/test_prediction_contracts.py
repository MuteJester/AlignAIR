"""Prediction contracts and matching external predictions to benchmark cases."""
import pytest
from alignair_benchmark import (
    GeneTruth,
    PredictionValidationAccumulator,
    align_predictions_to_cases,
    build_benchmark_report,
    case_to_prediction,
    generate_benchmark,
    prediction_contract,
    validate_prediction,
    validate_predictions,
    validate_predictions_for_cases,
)
from tests.alignair_benchmark._support import (_manual_case, _small_spec)

pytestmark = pytest.mark.integration
pytest.importorskip("GenAIRR")


def test_prediction_matching_recovers_reordered_outputs():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = build_benchmark_report(
        cases,
        list(reversed(preds)),
        match_by="sequence_id",
        contract_level="minimal",
    )
    assert report["prediction_matching"]["n_matched_cases"] == len(cases)
    assert report["prediction_matching"]["match_fraction"] == 1.0
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0


def test_prediction_matching_reports_missing_extra_and_duplicate_ids():
    cases = generate_benchmark(_small_spec())[:3]
    pred0 = case_to_prediction(cases[0])
    pred1 = case_to_prediction(cases[1])
    duplicate0 = dict(pred0)
    duplicate0["v_call"] = "not-a-real-allele"
    extra = dict(pred0)
    extra["sequence_id"] = "not-in-benchmark"

    result = align_predictions_to_cases(
        cases,
        [extra, duplicate0, pred1, pred0],
        id_field="sequence_id",
        duplicate_policy="first",
    )
    assert result.report["n_matched_cases"] == 2
    assert result.report["n_missing_cases"] == 1
    assert result.report["missing_case_ids"] == [cases[2].case_id]
    assert result.report["n_extra_prediction_ids"] == 1
    assert result.report["extra_prediction_ids"] == ["not-in-benchmark"]
    assert result.report["n_duplicate_prediction_ids"] == 1
    assert result.report["duplicate_prediction_ids"][0]["id"] == cases[0].case_id
    assert result.predictions[2] == {}

    report = build_benchmark_report(cases, [extra, duplicate0, pred1, pred0], match_by="sequence_id")
    assert report["prediction_matching"]["n_missing_cases"] == 1
    assert report["results"]["overall"]["genes"]["v"]["missing_call_rate"] > 0.0


def test_prediction_contract_validates_core_predictions():
    case = generate_benchmark(_small_spec())[0]
    pred = case_to_prediction(case)
    fields = prediction_contract()
    assert any(field["name"] == "v_call" and field["level"] == "minimal" for field in fields)
    assert any(field["name"] == "region_labels" and field["level"] == "assay" for field in fields)
    result = validate_prediction(pred, level="core", has_d=True)
    assert result["valid"]
    assert result["coverage_fraction"] == 1.0

    malformed = dict(pred)
    malformed["orientation_id"] = 9
    malformed.pop("v_sequence_start")
    result = validate_prediction(malformed, level="core", has_d=True)
    assert not result["valid"]
    assert "v_sequence_start" in result["missing_fields"]
    assert any(item["field"] == "orientation_id" for item in result["malformed_fields"])


def test_prediction_contract_batch_and_accumulator():
    case = generate_benchmark(_small_spec())[0]
    valid = case_to_prediction(case)
    invalid = {}
    batch = validate_predictions([valid, invalid], level="minimal", has_d=True)
    assert batch["n_predictions"] == 2
    assert batch["n_valid"] == 1
    assert batch["missing_field_counts"]["v_call"] == 1

    acc = PredictionValidationAccumulator(level="minimal", has_d=True)
    acc.update([valid])
    acc.update([invalid])
    summary = acc.to_dict()
    assert summary["n_predictions"] == 2
    assert summary["n_valid"] == 1
    assert summary["missing_field_counts"]["j_call"] == 1


def test_prediction_contract_can_validate_d_requirement_per_case():
    no_d_case = _manual_case("no_d", ("IGHV1-1*01",))
    no_d_case.scalars["productive"] = 1.0
    pred = case_to_prediction(no_d_case)

    case_aware = validate_predictions_for_cases([no_d_case], [pred], level="core")
    dataset_level = validate_predictions([pred], level="core", has_d=True)

    assert case_aware["case_aware"] is True
    assert case_aware["has_d"] == "per_case"
    assert case_aware["has_d_counts"] == {"has_d": 0, "no_d": 1}
    assert case_aware["valid_fraction"] == 1.0
    assert "d_call" not in case_aware["missing_field_counts"]
    assert dataset_level["valid_fraction"] == 0.0
    assert dataset_level["missing_field_counts"]["d_call"] == 1


def test_benchmark_report_uses_case_aware_contract_validation_for_mixed_chains():
    has_d_case = _manual_case("has_d", ("IGHV1-1*01",))
    has_d_case.genes["d"] = GeneTruth(("IGHD1-1*01",), "IGHD1-1*01", 80, 90, 0, 10)
    has_d_case.presented_genes["d"] = has_d_case.genes["d"]
    has_d_case.scalars["productive"] = 1.0
    no_d_case = _manual_case("no_d", ("IGHV1-2*01",))
    no_d_case.scalars["productive"] = 1.0
    cases = [has_d_case, no_d_case]
    preds = [case_to_prediction(case) for case in cases]

    report = build_benchmark_report(cases, preds, contract_level="core")

    validation = report["prediction_validation"]
    assert validation["case_aware"] is True
    assert validation["has_d"] == "per_case"
    assert validation["has_d_counts"] == {"has_d": 1, "no_d": 1}
    assert validation["valid_fraction"] == 1.0
    assert "d_call" not in validation["missing_field_counts"]
