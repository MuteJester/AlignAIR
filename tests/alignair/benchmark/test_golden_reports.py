import json
import os
from pathlib import Path

from alignair.benchmark import (
    BenchmarkCase,
    GeneTruth,
    ReadinessThresholds,
    assess_benchmark_readiness,
    build_benchmark_report,
    build_model_comparison_report,
    case_to_prediction,
)


GOLDEN_DIR = Path(__file__).with_name("golden")
UPDATE_GOLDENS = os.environ.get("ALIGNAIR_UPDATE_GOLDENS") == "1"


def _manual_case(case_id: str, v_call: str, *, stratum: str) -> BenchmarkCase:
    genes = {
        "v": GeneTruth((v_call,), v_call, 0, 80, 0, 80),
        "d": GeneTruth(),
        "j": GeneTruth(("IGHJ4*01",), "IGHJ4*01", 80, 100, 0, 20),
    }
    return BenchmarkCase(
        case_id=case_id,
        stratum=stratum,
        sequence="A" * 100,
        canonical_sequence="A" * 100,
        orientation_id=0,
        genes=genes,
        presented_genes=genes,
        scalars={
            "productive": 1.0,
            "mutation_rate": 0.02,
            "noise_count": 0.0,
            "indel_count": 0.0,
        },
        record={
            "locus": "IGH",
            "productive": True,
            "vj_in_frame": True,
            "stop_codon": False,
            "junction": "AACCGG",
            "junction_aa": "NG",
            "junction_start": 80,
            "junction_end": 86,
            "junction_length": 6,
            "np1": "CC",
            "np2": "GG",
            "np1_length": 2,
            "np2_length": 2,
            "p_v_3_length": 0,
            "p_d_5_length": 0,
            "p_d_3_length": 0,
            "p_j_5_length": 0,
            "d_inverted": False,
            "read_layout": "single",
        },
    )


def _golden_cases() -> list[BenchmarkCase]:
    return [
        _manual_case("golden-1", "IGHV1-1*01", stratum="easy"),
        _manual_case("golden-2", "IGHV1-2*01", stratum="hard"),
    ]


def _pick(mapping: dict, keys: tuple[str, ...]) -> dict:
    return {key: mapping[key] for key in keys if key in mapping}


def _assert_matches_golden(name: str, payload: dict) -> None:
    path = GOLDEN_DIR / name
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if UPDATE_GOLDENS:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert normalized == expected


def _report_snapshot(report: dict) -> dict:
    overall = report["results"]["overall"]
    return {
        "benchmark": report["benchmark"],
        "coverage": report["coverage"],
        "frame": report["frame"],
        "global": _pick(
            overall["global"],
            (
                "required_field_presence",
                "optional_field_presence",
                "parseable_airr_rate",
                "orientation_acc",
                "productive_acc",
                "vj_in_frame_acc",
                "stop_codon_acc",
                "junction_nt_exact",
                "junction_aa_exact",
                "junction_start_mae",
                "junction_end_mae",
                "junction_length_mae",
                "np1_exact",
                "np2_exact",
                "n1_length_mae",
                "n2_length_mae",
                "p_region_length_mae",
                "mutation_rate_mae",
                "noise_count_mae",
                "indel_count_mae",
            ),
        ),
        "genes": {
            gene: _pick(
                overall["genes"][gene],
                (
                    "found_rate",
                    "missing_call_rate",
                    "call_top1_in_set",
                    "gene_top1_in_set",
                    "call_set_precision",
                    "call_set_recall",
                    "call_set_f1",
                    "call_exact_set",
                    "ss_mae",
                    "se_mae",
                    "gs_mae",
                    "ge_mae",
                    "seq_span_iou",
                    "coordinate_parse_rate",
                    "missing_coordinate_rate",
                    "segment_length_mae",
                ),
            )
            for gene in ("v", "j")
        },
        "prediction_validation": _pick(
            report.get("prediction_validation", {}),
            (
                "level",
                "has_d",
                "case_aware",
                "has_d_counts",
                "n_predictions",
                "n_valid",
                "valid_fraction",
                "mean_coverage_fraction",
                "missing_field_counts",
                "malformed_field_counts",
            ),
        ),
        "criteria_audit_summary": _pick(
            report["criteria_audit"]["summary"],
            (
                "n_criteria",
                "status_counts",
                "n_observed_metric_keys",
                "n_metric_keys_without_criteria",
                "n_available_but_unobserved",
                "n_available_truth_field_gaps",
                "has_case_truth_audit",
            ),
        ),
        "assay_summary": _pick(
            report["assay"]["summary"],
            (
                "grade",
                "grade_counts",
                "completeness_gate_grade",
                "n_criteria",
                "n_criteria_with_results",
                "n_failed_criteria",
                "n_warned_criteria",
                "n_blocking_unscored_core_criteria",
                "n_truth_unavailable_core_criteria",
            ),
        ),
        "allele_diagnostics_v": _pick(
            report["diagnostics"]["allele_calling"]["genes"]["v"]["summary"],
            (
                "n_truth_cases",
                "top1_accepted_allele_rate",
                "top1_same_gene_rate",
                "top1_same_family_rate",
                "exact_set_rate",
                "wrong_family_rate",
                "same_gene_wrong_allele_rate",
                "same_family_wrong_gene_rate",
                "overcall_rate",
                "undercall_rate",
                "missing_prediction_rate",
            ),
        ),
        "boundary_diagnostics_v": _pick(
            report["diagnostics"]["boundaries"]["genes"]["v"]["summary"],
            (
                "n_truth_segments",
                "exact_all_coordinates_rate",
                "exact_query_span_rate",
                "exact_germline_span_rate",
                "query_coordinate_parse_rate",
                "germline_coordinate_parse_rate",
                "missing_coordinates_rate",
                "sequence_start_mae",
                "sequence_end_mae",
                "germline_start_mae",
                "germline_end_mae",
                "wrong_length_rate",
            ),
        ),
    }


def _readiness_snapshot(report: dict) -> dict:
    return {
        "grade": report["grade"],
        "profile": report["profile"],
        "n_cases": report["n_cases"],
        "grade_counts": report["grade_counts"],
        "coverage": report["coverage"],
        "checks": [
            _pick(check, ("name", "grade", "observed", "threshold"))
            for check in report["checks"]
        ],
    }


def _comparison_snapshot(report: dict) -> dict:
    return {
        "summary": report["summary"],
        "decision": {
            "verdict": report["decision"]["verdict"],
            "status_counts": report["decision"]["status_counts"],
            "primary_metrics": report["decision"]["primary_metrics"],
            "guardrail_metrics": report["decision"]["guardrail_metrics"],
            "primary_endpoints": [
                _pick(row, ("metric", "role", "status", "basis", "model_b_advantage", "reason"))
                for row in report["decision"]["primary_endpoints"]
            ],
            "guardrails": [
                _pick(row, ("metric", "role", "status", "basis", "model_b_advantage", "reason"))
                for row in report["decision"]["guardrails"]
            ],
            "multiple_comparison": report["decision"]["multiple_comparison"],
        },
        "overall": {
            metric: _pick(
                row,
                (
                    "metric",
                    "direction",
                    "model_a",
                    "model_b",
                    "raw_delta_model_b_minus_model_a",
                    "model_b_advantage",
                    "verdict",
                    "preferred_model",
                    "n_cases",
                    "n_compared_cases",
                    "win_loss_tie",
                ),
            )
            for metric, row in report["overall"].items()
        },
    }


def test_perfect_report_matches_golden_snapshot():
    cases = _golden_cases()
    predictions = [case_to_prediction(case) for case in cases]
    report = build_benchmark_report(cases, predictions, contract_level="core")
    _assert_matches_golden("perfect_report.json", _report_snapshot(report))


def test_broken_allele_report_matches_golden_snapshot():
    cases = _golden_cases()
    predictions = [case_to_prediction(case) for case in cases]
    for prediction in predictions:
        prediction["v_call"] = "IGHV9-9*01"
        prediction["v_calls"] = ["IGHV9-9*01"]
        prediction["junction"] = "TTTT"
    report = build_benchmark_report(cases, predictions, contract_level="core")
    _assert_matches_golden("broken_allele_report.json", _report_snapshot(report))


def test_readiness_report_matches_golden_snapshot():
    thresholds = ReadinessThresholds(
        profile="golden",
        min_cases=3,
        min_per_stratum=2,
        min_observed_alleles_per_gene=2,
        required_orientation_ids=(0, 1),
        min_per_orientation=1,
    )
    report = assess_benchmark_readiness(_golden_cases(), thresholds=thresholds)
    _assert_matches_golden("readiness_report.json", _readiness_snapshot(report))


def test_model_comparison_matches_golden_snapshot():
    cases = _golden_cases()
    predictions_a = [case_to_prediction(case) for case in cases]
    predictions_b = [case_to_prediction(case) for case in cases]
    for prediction in predictions_a:
        prediction["v_call"] = "IGHV9-9*01"
        prediction["v_calls"] = ["IGHV9-9*01"]
    for prediction in predictions_b:
        prediction["v_sequence_start"] += 2

    report = build_model_comparison_report(
        cases,
        predictions_a,
        predictions_b,
        model_a_name="old",
        model_b_name="new",
        metric_paths=("genes.v.call_top1_in_set", "genes.v.ss_mae"),
        primary_metrics=("genes.v.call_top1_in_set",),
        guardrail_metrics=("genes.v.ss_mae",),
        n_bootstrap=0,
        include_strata=False,
    )
    _assert_matches_golden("comparison_report.json", _comparison_snapshot(report))
