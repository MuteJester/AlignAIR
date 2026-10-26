"""Allele, annotation, coordinate, and uncertainty metrics."""
import pytest
from alignair_benchmark import (
    BenchmarkCase,
    DEFAULT_BOOTSTRAP_METRICS,
    GeneTruth,
    build_benchmark_report,
    bootstrap_metric_intervals,
    build_allele_calling_diagnostics,
    build_boundary_diagnostics,
    case_to_prediction,
    compact_summary,
    generate_benchmark,
    score_cases,
)
from tests.alignair_benchmark._support import (_manual_case, _small_spec)

pytestmark = pytest.mark.integration
pytest.importorskip("GenAIRR")


def test_perfect_predictions_score_high():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    summary = compact_summary(scores)
    assert summary["v"]["call"] == 1.0
    assert summary["v"]["set_f1"] == 1.0
    assert summary["v"]["seq_mae"] == [0.0, 0.0]
    assert summary["global"]["region_acc"] == 1.0
    assert summary["global"]["state_acc"] == 1.0
    assert summary["global"]["orientation_acc"] == 1.0
    assert summary["global"]["required_field_presence"] == 1.0
    assert summary["global"]["parseable_airr_rate"] == 1.0
    assert summary["global"]["junction_nt_exact"] == 1.0
    assert summary["global"]["junction_start_mae"] == 0.0
    assert summary["global"]["np1_exact"] == 1.0
    assert summary["global"]["p_region_length_mae"] == 0.0
    assert summary["global"]["vj_in_frame_acc"] == 1.0
    assert summary["global"]["stop_codon_acc"] == 1.0
    assert summary["global"]["d_inversion_acc"] == 1.0
    assert summary["global"]["region_macro_f1"] == 1.0
    assert summary["global"]["state_macro_f1"] == 1.0
    assert scores["genes"]["v"]["cigar_exact"] == 1.0
    assert scores["genes"]["v"]["cigar_edit_distance"] == 0.0
    assert scores["genes"]["v"]["trim_5_mae"] == 0.0
    assert scores["genes"]["v"]["identity_mae"] == 0.0
    assert "top10_recall" not in scores["genes"]["v"]
    assert set(scores["by_stratum"]) == {"clean", "oriented"}


def test_prediction_performance_fields_are_scored_and_reported():
    case = _manual_case("perf", ("IGHV1-2*01",))
    pred = case_to_prediction(case)
    pred.update(
        {
            "runtime_ms": 20.0,
            "peak_memory_mb": 123.0,
            "peak_memory_delta_mb": 7.5,
            "candidate_count": 42,
            "rerank_count": 5,
        }
    )

    scores = score_cases([case], [pred])
    global_scores = scores["global"]
    assert global_scores["milliseconds_per_read"] == 20.0
    assert global_scores["seconds_per_read"] == pytest.approx(0.02)
    assert global_scores["reads_per_second"] == pytest.approx(50.0)
    assert global_scores["peak_memory_mb"] == 123.0
    assert global_scores["peak_memory_delta_mb"] == 7.5
    assert global_scores["candidate_count"] == 42.0
    assert global_scores["rerank_count"] == 5.0

    report = build_benchmark_report([case], [pred])
    assert report["performance"]["source"] == "prediction_fields"
    assert report["performance"]["wall_time_seconds"] == pytest.approx(0.02)
    assert report["performance"]["peak_memory_mb"] == 123.0
    assert report["assay"]["by_category"]["efficiency"]["n_with_results"] == 1


def test_bootstrap_uncertainty_uses_existing_genairr_truth():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = bootstrap_metric_intervals(cases, preds, n_bootstrap=20, confidence=0.9, seed=11)
    assert "genes.v.call_top1_in_set" in DEFAULT_BOOTSTRAP_METRICS
    assert report["method"] == "paired_case_bootstrap"
    assert report["truth_source"] == "GenAIRR benchmark cases"
    assert report["n_bootstrap"] == 20
    assert report["confidence"] == 0.9
    metric = report["overall"]["genes.v.call_top1_in_set"]
    assert metric["point"] == 1.0
    assert metric["ci_low"] == 1.0
    assert metric["ci_high"] == 1.0
    assert report["by_stratum"]["clean"]["genes.v.call_top1_in_set"]["n_cases"] == 3


def test_set_metrics_reward_set_outputs():
    cases = generate_benchmark(_small_spec())
    case = cases[0]
    pred = case_to_prediction(case)
    truth = case.genes["v"]
    if len(truth.calls) == 1:
        pred["v_calls"] = [truth.calls[0], "not-a-real-allele"]
        scores = score_cases([case], [pred])
        assert scores["genes"]["v"]["call_set_recall"] == 1.0
        assert scores["genes"]["v"]["call_set_precision"] == 0.5


def test_topk_candidate_metrics_require_explicit_ranked_outputs():
    case = _manual_case("topk", ("IGHV1-2*01",))
    base_pred = {
        "v_call": "IGHV1-2*01",
        "v_calls": ["IGHV1-2*01"],
        "j_call": "IGHJ4*01",
        "j_calls": ["IGHJ4*01"],
    }

    no_candidate_scores = score_cases([case], [base_pred])
    assert "top1_recall" not in no_candidate_scores["genes"]["v"]
    assert "top10_recall" not in no_candidate_scores["genes"]["v"]

    ranked_pred = {
        **base_pred,
        "v_ranked_calls": ["IGHV9-9*01", "IGHV1-2*01", "IGHV3-7*01"],
    }
    ranked_scores = score_cases([case], [ranked_pred])["genes"]["v"]
    assert ranked_scores["top1_recall"] == 0.0
    assert ranked_scores["top3_recall"] == 1.0
    assert ranked_scores["top10_recall"] == 1.0
    assert ranked_scores["topk_truth_set_recall"] == 1.0
    assert ranked_scores["same_gene_sibling_top1"] == 0.0
    assert ranked_scores["sibling_set_recall"] == 1.0

    score_pred = {
        **base_pred,
        "v_scores": {
            "IGHV9-9*01": 0.9,
            "IGHV1-2*01": 0.8,
        },
    }
    score_scores = score_cases([case], [score_pred])["genes"]["v"]
    assert score_scores["top1_recall"] == 0.0
    assert score_scores["top3_recall"] == 1.0

    topk_alias_pred = {
        **base_pred,
        "v_topk": ["IGHV1-2*01"],
    }
    alias_scores = score_cases([case], [topk_alias_pred])["genes"]["v"]
    assert alias_scores["top1_recall"] == 1.0


def test_contaminant_handling_credits_flag_only():
    # a true contaminant; the model FLAGS it (is_contaminant=True) but keeps calls (flag-only).
    genes = {
        "v": GeneTruth(calls=("IGHV1-2*01",), primary="IGHV1-2*01",
                       sequence_start=0, sequence_end=80, germline_start=0, germline_end=80),
        "d": GeneTruth(),
        "j": GeneTruth(calls=("IGHJ4*01",), primary="IGHJ4*01",
                       sequence_start=80, sequence_end=100, germline_start=0, germline_end=20),
    }
    case = BenchmarkCase(case_id="c", stratum="contaminant", sequence="A" * 100,
                         canonical_sequence="A" * 100, orientation_id=0, genes=genes,
                         presented_genes=genes, record={"is_contaminant": True})
    base = {"v_call": "IGHV1-2*01", "v_calls": ["IGHV1-2*01"],
            "j_call": "IGHJ4*01", "j_calls": ["IGHJ4*01"]}
    g = score_cases([case], [{**base, "is_contaminant": True}])["global"]
    assert g["contaminant_handled_rate"] == 1.0           # flag-only counts as handled
    assert g["contaminant_flag_acc"] == 1.0
    assert g["false_positive_alignment_rate"] == 0.0      # flagged -> not a false positive
    # a confident call that is NOT flagged = false positive, not handled
    g2 = score_cases([case], [{**base, "is_contaminant": False}])["global"]
    assert g2["contaminant_handled_rate"] == 0.0 and g2["false_positive_alignment_rate"] == 1.0


def test_graceful_degradation_metrics():
    case = _manual_case("g", ("IGHV1-2*01",))                  # truth gene IGHV1-2, family IGHV1

    def gv(resolved, level, call="IGHV1-2*01"):
        pred = {"v_call": call, "v_calls": [call], "v_resolved_call": resolved,
                "v_call_level": level, "j_call": "IGHJ4*01", "j_calls": ["IGHJ4*01"]}
        return score_cases([case], [pred])["genes"]["v"]

    g = gv("IGHV1-2", "gene")                                  # correct coarser call
    assert g["graceful_useful"] == 1.0 and g["graceful_hard_error"] == 0.0 and g["graceful_non_error"] == 1.0
    g = gv("IGHV1", "family")                                  # correct family
    assert g["graceful_useful"] == 1.0 and g["graceful_hard_error"] == 0.0
    g = gv(None, "none")                                       # honest abstention is not an error
    assert g["graceful_abstain"] == 1.0 and g["graceful_hard_error"] == 0.0 and g["graceful_non_error"] == 1.0
    g = gv("IGHV3-23", "gene", call="IGHV3-23*01")             # confidently wrong -> hard error
    assert g["graceful_hard_error"] == 1.0 and g["graceful_non_error"] == 0.0 and g["graceful_useful"] == 0.0


def test_allele_calling_diagnostics_separate_allele_gene_and_family_errors():
    cases = [
        _manual_case("exact", ("IGHV1-1*01",)),
        _manual_case("same_gene_wrong_allele", ("IGHV1-1*02",)),
        _manual_case("same_family_wrong_gene", ("IGHV1-2*01",)),
        _manual_case("wrong_family", ("IGHV3-7*01",)),
        _manual_case("ambiguous", ("IGHV2-5*01", "IGHV2-5*02")),
        _manual_case("missing", ("IGHV5-10*01",)),
    ]
    preds = [
        {"v_call": "IGHV1-1*01", "v_calls": ["IGHV1-1*01"]},
        {"v_call": "IGHV1-1*01", "v_calls": ["IGHV1-1*01"]},
        {"v_call": "IGHV1-69*01", "v_calls": ["IGHV1-69*01"]},
        {"v_call": "IGHV4-4*01", "v_calls": ["IGHV4-4*01"]},
        {"v_call": "IGHV2-5*02", "v_calls": ["IGHV2-5*02"]},
        {},
    ]

    diagnostics = build_allele_calling_diagnostics(cases, preds, top_n=20)
    v_diag = diagnostics["genes"]["v"]
    summary = v_diag["summary"]

    assert summary["n_truth_cases"] == 6
    assert summary["top1_accepted_allele_rate"] == pytest.approx(2 / 6)
    assert summary["top1_same_gene_rate"] == pytest.approx(3 / 6)
    assert summary["top1_same_family_rate"] == pytest.approx(4 / 6)
    assert summary["same_gene_wrong_allele_rate"] == pytest.approx(1 / 6)
    assert summary["same_family_wrong_gene_rate"] == pytest.approx(1 / 6)
    assert summary["wrong_family_rate"] == pytest.approx(1 / 6)
    assert summary["missing_prediction_rate"] == pytest.approx(1 / 6)

    allele_rows = {row["allele"]: row for row in v_diag["per_allele"]}
    assert allele_rows["IGHV1-1*02"]["singleton_top1_exact_rate"] == 0.0
    assert allele_rows["IGHV1-1*02"]["same_gene_wrong_allele_rate"] == 1.0
    assert allele_rows["IGHV2-5*01"]["n_ambiguous_truth_cases"] == 1
    assert allele_rows["IGHV2-5*01"]["top1_accepted_allele_rate"] == 1.0
    assert allele_rows["IGHV2-5*01"]["pred_set_contains_allele_rate"] == 0.0
    assert allele_rows["IGHV5-10*01"]["missing_prediction_rate"] == 1.0

    gene_rows = {row["gene"]: row for row in v_diag["per_gene"]}
    assert gene_rows["IGHV1-2"]["same_family_wrong_gene_rate"] == 1.0
    family_rows = {row["family"]: row for row in v_diag["per_gene_family"]}
    assert family_rows["IGHV1"]["top1_same_family_rate"] == 1.0
    assert family_rows["IGHV3"]["wrong_family_rate"] == 1.0
    assert family_rows["IGHV5"]["missing_prediction_rate"] == 1.0

    allele_confusions = {
        (row["truth_allele"], row["pred_call"]): row for row in v_diag["allele_confusions"]
    }
    assert allele_confusions[("IGHV1-1*02", "IGHV1-1*01")]["error_kind"] == "same_gene_wrong_allele"
    assert allele_confusions[("IGHV1-2*01", "IGHV1-69*01")]["error_kind"] == "same_family_wrong_gene"
    assert allele_confusions[("IGHV3-7*01", "IGHV4-4*01")]["error_kind"] == "wrong_family"
    assert allele_confusions[("IGHV5-10*01", None)]["error_kind"] == "missing_prediction"
    assert "same_gene_wrong_allele" in allele_confusions[("IGHV1-1*02", "IGHV1-1*01")]["example_case_ids"]


def test_boundary_diagnostics_classify_coordinate_failure_modes():
    cases = [
        _manual_case("exact", ("IGHV1-1*01",)),
        _manual_case("missing", ("IGHV1-1*01",)),
        _manual_case("start_only", ("IGHV1-1*01",)),
        _manual_case("shifted", ("IGHV1-1*01",)),
        _manual_case("wrong_length", ("IGHV1-1*01",)),
        _manual_case("negative", ("IGHV1-1*01",)),
        _manual_case("wrong_trim", ("IGHV1-1*01",)),
    ]

    def pred(start, end, gs=0, ge=80):
        return {
            "v_call": "IGHV1-1*01",
            "v_sequence_start": start,
            "v_sequence_end": end,
            "v_germline_start": gs,
            "v_germline_end": ge,
        }

    preds = [
        pred(0, 80),
        {},
        pred(1, 80),
        pred(1, 81),
        pred(0, 70),
        pred(80, 0),
        pred(0, 80, gs=5, ge=85),
    ]

    diagnostics = build_boundary_diagnostics(cases, preds, top_n=20)
    v_diag = diagnostics["genes"]["v"]
    summary = v_diag["summary"]

    assert summary["n_truth_segments"] == 7
    assert summary["exact_query_span_rate"] == pytest.approx(2 / 7)
    assert summary["exact_all_coordinates_rate"] == pytest.approx(1 / 7)
    assert summary["missing_coordinates_rate"] == pytest.approx(1 / 7)
    assert summary["start_only_error_rate"] == pytest.approx(1 / 7)
    assert summary["off_by_one_rate"] == pytest.approx(2 / 7)
    assert summary["systematic_plus_one_shift_rate"] == pytest.approx(1 / 7)
    assert summary["correct_length_shifted_span_rate"] == pytest.approx(1 / 7)
    assert summary["wrong_length_rate"] == pytest.approx(3 / 7)
    assert summary["negative_span_rate"] == pytest.approx(1 / 7)
    assert summary["correct_query_span_wrong_germline_span_rate"] == pytest.approx(1 / 7)
    assert summary["correct_allele_wrong_trim_rate"] == pytest.approx(1 / 7)

    failures = {row["failure_type"]: row for row in v_diag["failure_types"]}
    assert failures["missing_coordinates"]["example_case_ids"] == ["missing"]
    assert failures["systematic_plus_one_shift"]["example_case_ids"] == ["shifted"]
    assert failures["correct_allele_wrong_trim"]["example_case_ids"] == ["wrong_trim"]


def test_boundary_diagnostics_detect_frame_confusion_and_vdj_overlap():
    canonical_genes = {
        "v": GeneTruth(("IGHV1-1*01",), "IGHV1-1*01", 0, 80, 0, 80),
        "d": GeneTruth(),
        "j": GeneTruth(("IGHJ4*01",), "IGHJ4*01", 80, 100, 0, 20),
    }
    presented_genes = {
        "v": GeneTruth(("IGHV1-1*01",), "IGHV1-1*01", 20, 100, 0, 80),
        "d": GeneTruth(),
        "j": GeneTruth(("IGHJ4*01",), "IGHJ4*01", 0, 20, 0, 20),
    }
    frame_case = BenchmarkCase(
        case_id="frame_confusion",
        stratum="manual",
        sequence="A" * 100,
        canonical_sequence="A" * 100,
        orientation_id=3,
        genes=canonical_genes,
        presented_genes=presented_genes,
        record={},
    )
    overlap_case = _manual_case("overlap", ("IGHV1-1*01",))
    preds = [
        {
            "v_sequence_start": 20,
            "v_sequence_end": 100,
            "v_germline_start": 0,
            "v_germline_end": 80,
        },
        {
            "v_sequence_start": 0,
            "v_sequence_end": 90,
            "v_germline_start": 0,
            "v_germline_end": 80,
            "j_sequence_start": 80,
            "j_sequence_end": 100,
            "j_germline_start": 0,
            "j_germline_end": 20,
        },
    ]

    diagnostics = build_boundary_diagnostics([frame_case, overlap_case], preds, top_n=20)
    v_summary = diagnostics["genes"]["v"]["summary"]
    assert v_summary["canonical_presented_frame_confusion_rate"] == pytest.approx(0.5)
    failures = {row["failure_type"]: row for row in diagnostics["global"]["failure_types"]}
    assert failures["vdj_order_missing_coordinates"]["example_case_ids"] == ["frame_confusion"]
    assert failures["vdj_order_or_overlap_error"]["example_case_ids"] == ["overlap"]


def test_extended_metrics_detect_biological_annotation_errors():
    case = generate_benchmark(_small_spec())[0]
    pred = case_to_prediction(case)
    pred["junction"] = "A"
    pred["np1"] = "A"
    pred["v_cigar"] = "1M"
    pred["vj_in_frame"] = not bool(case.record["vj_in_frame"])
    pred["d_inverted"] = not bool(case.record["d_inverted"])
    scores = score_cases([case], [pred])
    assert scores["global"]["junction_nt_exact"] == 0.0
    assert scores["global"]["np1_exact"] == 0.0
    assert scores["global"]["vj_in_frame_acc"] == 0.0
    assert scores["global"]["d_inversion_acc"] == 0.0
    assert scores["genes"]["v"]["cigar_exact"] == 0.0
    assert scores["genes"]["v"]["cigar_edit_distance"] > 0.0
    empty_scores = score_cases([case], [{}])
    assert empty_scores["global"]["required_field_presence"] == 0.0
    assert empty_scores["global"]["parseable_airr_rate"] == 0.0
