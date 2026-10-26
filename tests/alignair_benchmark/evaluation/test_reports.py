"""Assay reports, completeness gates, and online evaluation."""
import pytest
from alignair_benchmark import (
    audit_criteria_report,
    build_assay_report,
    build_benchmark_report,
    case_to_prediction,
    generate_benchmark,
    run_benchmark_report,
    run_online_benchmark,
    score_cases,
    stream_benchmark,
)
from tests.alignair_benchmark._support import (_manual_case, _small_spec)

pytestmark = pytest.mark.integration
pytest.importorskip("GenAIRR")


def test_benchmark_report_includes_allele_diagnostics():
    cases = [
        _manual_case("ok", ("IGHV1-1*01",)),
        _manual_case("bad", ("IGHV3-7*01",)),
    ]
    preds = [
        {"v_call": "IGHV1-1*01", "v_calls": ["IGHV1-1*01"]},
        {"v_call": "IGHV4-4*01", "v_calls": ["IGHV4-4*01"]},
    ]

    report = build_benchmark_report(cases, preds)
    v_diag = report["diagnostics"]["allele_calling"]["genes"]["v"]
    assert v_diag["summary"]["n_truth_cases"] == 2
    assert v_diag["summary"]["wrong_family_rate"] == 0.5
    assert v_diag["family_confusions"][0]["truth_family"] == "IGHV3"
    assert report["diagnostics"]["boundaries"]["genes"]["v"]["summary"]["n_truth_segments"] == 2


def test_assay_report_groups_criteria_and_missing_metrics():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    report = build_assay_report(scores, top_n_contexts=5)
    assert report["summary"]["n_cases"] == len(cases)
    assert report["summary"]["n_criteria_with_results"] > 0
    assert report["summary"]["grade"] == "pass"
    assert report["summary"]["n_failed_criteria"] == 0
    assert "allele_calling" in report["by_category"]
    assert "segmentation" in report["by_category"]
    assert report["by_category"]["allele_calling"]["grade"] == "pass"
    by_name = {entry["name"]: entry for entry in report["criteria"]}
    assert by_name["allele_top1_call"]["observed_metrics"]["call_top1_in_set"]["genes"]["v"] == 1.0
    assert by_name["allele_top1_call"]["grade"] == "pass"
    assert by_name["allele_top1_call"]["metric_assessments"][0]["grade"] == "pass"
    assert by_name["allele_uncertainty_calibration"]["n_observed_metric_keys"] == 0
    assert by_name["allele_uncertainty_calibration"]["grade"] == "planned"


def test_assay_report_surfaces_weak_contexts():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    first_stratum = next(iter(scores["by_stratum"]))
    scores["by_stratum"][first_stratum]["genes"]["v"]["call_top1_in_set"] = 0.0
    report = build_assay_report(scores, top_n_contexts=3)
    assert report["weak_contexts"]
    assert report["weak_contexts"][0]["metric"] == "call_top1_in_set"
    assert report["weak_contexts"][0]["context"] == first_stratum


def test_assay_report_grades_broken_core_criteria_as_failures():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    for pred in preds:
        pred["v_call"] = "not-a-real-allele"
        pred["v_calls"] = ["not-a-real-allele"]
    report = build_assay_report(score_cases(cases, preds), top_n_contexts=5)
    assert report["summary"]["grade"] == "fail"
    assert report["summary"]["n_failed_criteria"] > 0
    by_name = {entry["name"]: entry for entry in report["criteria"]}
    assert by_name["allele_top1_call"]["grade"] == "fail"
    assert any(row["grade"] == "fail" for row in by_name["allele_top1_call"]["metric_assessments"])
    assert any(item["name"] == "allele_top1_call" for item in report["critical_failures"])


def test_criteria_audit_tracks_metric_and_genairr_truth_coverage():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    audit = audit_criteria_report(scores, cases=cases)
    assert audit["summary"]["has_case_truth_audit"]
    assert audit["truth_source"] == "GenAIRR benchmark cases"
    assert "call_top1_in_set" in audit["observed_metric_keys"]["all"]
    assert "truth_set_size" not in audit["metric_keys_without_criteria"]
    assert "ss_within10" not in audit["metric_keys_without_criteria"]
    by_name = {entry["name"]: entry for entry in audit["criteria"]}
    assert by_name["allele_top1_call"]["metric_coverage_fraction"] == 1.0
    assert "call_top1_in_set" in by_name["allele_top1_call"]["observed_metric_keys"]
    assert audit["truth_field_availability"]["v_call"]["n_present"] == len(cases)
    assert audit["truth_field_availability"]["sequence"]["n_present"] == len(cases)


def test_build_benchmark_report_from_saved_predictions():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = build_benchmark_report(
        cases,
        preds,
        contract_level="core",
        n_bootstrap=10,
        confidence=0.9,
        bootstrap_strata=False,
    )
    assert report["coverage"]["n_cases"] == len(cases)
    assert report["catalog_validation"]["valid"] is True
    assert report["catalog_validation"]["summary"]["n_unmapped_contexts"] == 0
    assert report["prediction_validation"]["valid_fraction"] == 1.0
    assert report["criteria_audit"]["summary"]["has_case_truth_audit"]
    assert report["criteria_audit"]["truth_field_availability"]["v_call"]["n_present"] == len(cases)
    assert report["uncertainty"]["overall"]["genes.v.call_top1_in_set"]["ci_low"] == 1.0
    assert report["uncertainty"]["confidence"] == 0.9
    assert report["uncertainty"]["by_stratum"] == {}
    assert report["assay"]["summary"]["n_criteria_with_results"] > 0
    assert report["assay"]["summary"]["grade"] == "pass"
    assert report["assay"]["summary"]["completeness_gate_grade"] == "pass"
    assert report["assay"]["summary"]["n_blocking_unscored_core_criteria"] == 0
    assert any(
        row["name"] == "constant_region_call"
        for row in report["assay"]["completeness_gate"]["truth_unavailable_core_criteria"]
    )
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0
    assert "stratum:clean" in report["results"]["by_context"]
    assert any(k.startswith("orientation:") for k in report["results"]["by_context"])
    assert any(k.startswith("length:") for k in report["results"]["by_context"])
    assert any(k.startswith("mutation:") for k in report["results"]["by_context"])
    assert any(k.startswith("indel:") for k in report["results"]["by_context"])
    assert any(k.startswith("noise:") for k in report["results"]["by_context"])
    assert any(k.startswith("segment_presence:") for k in report["results"]["by_context"])
    assert any(k.startswith("ambiguity:") for k in report["results"]["by_context"])


def test_run_benchmark_report_from_predictor():
    cases = generate_benchmark(_small_spec())

    def predictor(reads):
        assert len(reads) == len(cases)
        return [case_to_prediction(c) for c in cases]

    report = run_benchmark_report(cases, predictor, contract_level="core")
    assert report["prediction_validation"]["n_valid"] == len(cases)
    assert report["performance"]["n_sequences"] == len(cases)
    assert report["performance"]["wall_time_seconds"] >= 0.0
    assert report["performance"]["reads_per_second"] > 0.0
    assert report["results"]["overall"]["global"]["milliseconds_per_read"] >= 0.0
    assert report["assay"]["by_category"]["allele_calling"]["n_with_results"] > 0


def test_assay_completeness_gate_fails_unscored_available_core_criteria():
    criteria = [
        {
            "category": "unit",
            "name": "required_core_missing",
            "metric_keys": ("missing_metric",),
            "description": "Synthetic required criterion for assay completeness.",
            "contexts": ("unit",),
            "importance": "core",
            "status": "available",
            "ground_truth_fields": ("sequence",),
        }
    ]
    payload = {
        "frame": "canonical",
        "criteria": criteria,
        "results": {
            "overall": {"n_cases": 1, "frame": "canonical", "global": {}, "genes": {}},
            "by_context": {},
        },
        "criteria_audit": {
            "summary": {"has_case_truth_audit": True},
            "criteria": [
                {
                    "name": "required_core_missing",
                    "category": "unit",
                    "status": "available",
                    "importance": "core",
                    "metric_keys": ("missing_metric",),
                    "observed_metric_keys": (),
                    "missing_metric_keys": ("missing_metric",),
                    "metric_coverage_fraction": 0.0,
                    "ground_truth_fields": ("sequence",),
                    "unavailable_truth_fields": (),
                }
            ],
        },
    }

    report = build_assay_report(payload)

    assert report["summary"]["grade"] == "fail"
    assert report["summary"]["completeness_gate_grade"] == "fail"
    assert report["summary"]["n_failed_criteria"] == 1
    assert report["summary"]["n_blocking_unscored_core_criteria"] == 1
    assert report["completeness_gate"]["blocking_unscored_core_criteria"][0]["name"] == "required_core_missing"
    by_name = {entry["name"]: entry for entry in report["criteria"]}
    assert by_name["required_core_missing"]["grade"] == "fail"
    assert by_name["required_core_missing"]["completeness_gate_failed"] is True


def test_online_benchmark_report_with_perfect_predictor():
    spec = _small_spec()

    def predictor(reads):
        # The online runner calls the predictor with generated reads only; for this
        # deterministic test, close over a matching stream of perfect cases.
        assert reads
        batch = [next(predictor.case_iter) for _ in reads]
        return [case_to_prediction(c) for c in batch]

    predictor.case_iter = stream_benchmark(spec)
    report = run_online_benchmark(spec, predictor, batch_size=2, contract_level="core")
    assert report["coverage"]["n_cases"] == spec.n_cases
    assert report["coverage"]["alleles"]["v"]["n_total_reference"] > 0
    assert 0.0 < report["coverage"]["alleles"]["v"]["fraction_observed"] <= 1.0
    assert report["criteria"]
    assert report["prediction_contract"]
    assert report["prediction_validation"]["valid_fraction"] == 1.0
    assert report["scenario_axes"]
    assert report["performance"]["n_sequences"] == spec.n_cases
    assert report["performance"]["n_batches"] == 4
    assert report["performance"]["reads_per_second"] > 0.0
    assert report["results"]["overall"]["global"]["seconds_per_read"] >= 0.0
    assert report["assay"]["summary"]["n_criteria_with_results"] > 0
    assert report["diagnostics"]["allele_calling"]["genes"]["v"]["summary"]["n_truth_cases"] == spec.n_cases
    assert report["diagnostics"]["boundaries"]["genes"]["v"]["summary"]["n_truth_segments"] == spec.n_cases
    assert "allele_calling" in report["assay"]["by_category"]
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0
    assert "stratum:clean" in report["results"]["by_context"]
    assert any(k.startswith("locus:") for k in report["results"]["by_context"])
    assert any(k.startswith("chain:") for k in report["results"]["by_context"])
    assert any(k.startswith("d_orientation:") for k in report["results"]["by_context"])
    assert any(k.startswith("junction_length:") for k in report["results"]["by_context"])
    assert any(k.startswith("segment_presence:") for k in report["results"]["by_context"])
    assert "ambiguity:all_single" in report["results"]["by_context"] or any(
        k.startswith("ambiguity:") for k in report["results"]["by_context"]
    )
