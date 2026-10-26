"""Benchmark CLI exports, evaluation, catalog queries, and reports."""
import pytest
import csv
import json
from alignair_benchmark import (
    BenchmarkSpec,
    StratumSpec,
    build_benchmark_report,
    case_to_prediction,
    criteria_catalog,
    generate_benchmark,
    load_dicts_jsonl,
    load_jsonl,
    save_jsonl,
    save_dicts_jsonl,
)
from alignair_benchmark.cli import main as benchmark_cli
from tests.alignair_benchmark._support import (
    _airr_row_for_case,
    _manual_case,
    _single_clean_case,
    _small_spec,
    _write_tsv,
)

pytestmark = pytest.mark.integration
pytest.importorskip("GenAIRR")


def test_cli_readiness_command_reports_preflight_status(tmp_path):
    cases = generate_benchmark(_small_spec())
    cases_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "readiness.json"
    save_jsonl(cases, cases_path)
    benchmark_cli(
        [
            "readiness",
            "--cases",
            str(cases_path),
            "--config",
            "HUMAN_IGH_OGRDB",
            "--profile",
            "smoke",
            "--out",
            str(out_path),
        ]
    )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["grade"] == "pass"
    assert report["profile"] == "smoke"
    assert report["n_cases"] == len(cases)


def test_cli_export_writes_external_tool_inputs(tmp_path):
    cases = generate_benchmark(_small_spec())
    cases_path = tmp_path / "cases.jsonl"
    out_dir = tmp_path / "exported"
    save_jsonl(cases, cases_path)

    benchmark_cli(
        [
            "export",
            "--cases",
            str(cases_path),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "cli",
            "--config",
            "HUMAN_IGH_OGRDB",
            "--airr-metadata",
        ]
    )
    assert (out_dir / "cli.fasta").exists()
    assert (out_dir / "cli_airr_input.tsv").exists()
    manifest = json.loads((out_dir / "cli_manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"]["n_cases"] == len(cases)
    assert manifest["reference"]["has_d"] is True
    with (out_dir / "cli_airr_input.tsv").open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["sequence_id"] == cases[0].case_id
    assert row["benchmark_case_id"] == cases[0].case_id


def test_cli_normalizes_and_evaluates_airr_tables(tmp_path):
    case = _single_clean_case(seed=37)
    cases_path = tmp_path / "cases.jsonl"
    airr_path = tmp_path / "airr.tsv"
    preds_path = tmp_path / "predictions.jsonl"
    performance_path = tmp_path / "performance.json"
    report_path = tmp_path / "report.json"
    save_jsonl([case], cases_path)
    airr_row = _airr_row_for_case(case)
    airr_row["runtime_ms"] = "10"
    airr_row["peak_memory_mb"] = "128"
    _write_tsv(airr_path, [airr_row])
    performance_path.write_text(
        json.dumps(
            {
                "wall_time_seconds": 2.0,
                "peak_memory_mb": 256.0,
                "candidate_count": 12,
            }
        ),
        encoding="utf-8",
    )

    benchmark_cli(
        [
            "normalize-predictions",
            "--input",
            str(airr_path),
            "--format",
            "airr-tsv",
            "--out",
            str(preds_path),
        ]
    )
    preds = load_dicts_jsonl(preds_path)
    assert preds[0]["j_call"] == case.genes["j"].calls[0]
    assert preds[0]["runtime_ms"] == "10"
    assert preds[0]["peak_memory_mb"] == "128"

    benchmark_cli(
        [
            "evaluate",
            "--cases",
            str(cases_path),
            "--predictions",
            str(airr_path),
            "--prediction-format",
            "airr-tsv",
            "--contract-level",
            "core",
            "--performance-json",
            str(performance_path),
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["prediction_validation"]["valid_fraction"] == 1.0
    assert report["results"]["overall"]["genes"]["j"]["call_top1_in_set"] == 1.0
    assert report["performance"]["seconds_per_read"] == 2.0
    assert report["performance"]["peak_memory_mb"] == 256.0
    assert report["results"]["overall"]["global"]["candidate_count"] == 12.0


def test_cli_evaluate_matches_airr_rows_by_sequence_id(tmp_path):
    cases = generate_benchmark(
        BenchmarkSpec(
            name="airr_match",
            dataconfig_name="HUMAN_IGH_OGRDB",
            seed=41,
            strata=(StratumSpec(name="clean", n=2, progress=0.0),),
        )
    )
    cases_path = tmp_path / "cases.jsonl"
    airr_path = tmp_path / "airr.tsv"
    report_path = tmp_path / "report.json"
    save_jsonl(cases, cases_path)
    _write_tsv(airr_path, [_airr_row_for_case(cases[1]), _airr_row_for_case(cases[0])])

    benchmark_cli(
        [
            "evaluate",
            "--cases",
            str(cases_path),
            "--predictions",
            str(airr_path),
            "--prediction-format",
            "airr-tsv",
            "--bootstrap",
            "5",
            "--confidence",
            "0.9",
            "--no-bootstrap-strata",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["prediction_matching"]["n_matched_cases"] == 2
    assert report["uncertainty"]["n_bootstrap"] == 5
    assert report["uncertainty"]["confidence"] == 0.9
    assert report["uncertainty"]["by_stratum"] == {}
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0


def test_cli_compare_matches_prediction_ids_and_writes_report(tmp_path):
    cases = [
        _manual_case("cli_cmp1", ("IGHV1-1*01",)),
        _manual_case("cli_cmp2", ("IGHV1-2*01",)),
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

    cases_path = tmp_path / "cases.jsonl"
    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    report_path = tmp_path / "comparison.json"
    save_jsonl(cases, cases_path)
    save_dicts_jsonl(list(reversed(preds_a)), a_path)
    save_dicts_jsonl(preds_b, b_path)

    benchmark_cli(
        [
            "compare",
            "--cases",
            str(cases_path),
            "--a-predictions",
            str(a_path),
            "--b-predictions",
            str(b_path),
            "--model-a-name",
            "old",
            "--model-b-name",
            "new",
            "--metric",
            "genes.v.call_top1_in_set",
            "--policy",
            "allele_calling_core",
            "--multiple-comparison-correction",
            "bonferroni",
            "--bootstrap",
            "5",
            "--confidence",
            "0.9",
            "--no-bootstrap-strata",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = report["overall"]["genes.v.call_top1_in_set"]
    assert report["prediction_matching"]["model_a"]["n_matched_cases"] == 2
    assert report["prediction_matching"]["model_b"]["n_matched_cases"] == 2
    assert report["summary"]["verdict"] == "model_b_better"
    assert report["comparison"]["comparison_policy"] == "allele_calling_core"
    assert report["decision"]["multiple_comparison"]["method"] == "bonferroni"
    assert report["decision"]["multiple_comparison"]["applied"] is True
    assert report["decision"]["verdict"] == "model_b_superior"
    assert all(row["status"] == "pass" for row in report["decision"]["primary_endpoints"])
    assert row["preferred_model"] == "new"
    assert row["bootstrap_probability_model_b_better"] == 1.0
    assert report["by_stratum"] == {}


def test_cli_lists_comparison_policy_templates(capsys):
    benchmark_cli(["comparison-policies"])
    payload = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in payload["comparison_policies"]}
    assert {"allele_calling_core", "igh_allele_calling_core", "boundary_core", "airr_core"} <= names


def test_cli_validates_comparison_policy_templates(tmp_path):
    out = tmp_path / "policy_validation.json"

    benchmark_cli(["validate-comparison-policies", "--out", str(out)])

    validation = json.loads(out.read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert validation["summary"]["n_invalid_policy_metrics"] == 0


def test_cli_build_reports_generation_profile_with_workers(tmp_path, capsys):
    out = tmp_path / "empty.jsonl"
    benchmark_cli(
        [
            "build",
            "--out",
            str(out),
            "--recipe",
            "broad",
            "--n-per-stratum",
            "0",
            "--workers",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out.rsplit("\nwrote", 1)[0])
    assert out.exists()
    assert payload["coverage"]["n_cases"] == 0
    assert payload["generation_profile"]["mode"] == "fixed_size"
    assert payload["generation_profile"]["workers"] == 2
    assert payload["generation_profile"]["parallel"] is True


def test_cli_lists_measurement_scenarios(capsys):
    benchmark_cli(["measurement-scenarios"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"]["valid"]
    assert any(row["name"] == "allele_coverage_and_candidates" for row in payload["measurement_scenarios"])


def test_cli_builds_restricted_allele_panel_recipe(tmp_path, capsys):
    out = tmp_path / "allele_panels.jsonl"
    benchmark_cli(
        [
            "build",
            "--recipe",
            "allele-panel",
            "--n-panels",
            "1",
            "--alleles-per-panel",
            "2",
            "--n-per-panel",
            "1",
            "--out",
            str(out),
        ]
    )

    payload = json.loads(capsys.readouterr().out.rsplit("\nwrote", 1)[0])
    cases = load_jsonl(out)
    assert payload["coverage"]["n_cases"] == 1
    assert payload["measurement_coverage"]["explicit_by_measurement"]["allele_coverage_and_candidates"] == 1
    assert cases[0].record["benchmark_measurement"] == "allele_coverage_and_candidates"


def test_cli_audit_command_writes_metric_and_truth_audit(tmp_path):
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = build_benchmark_report(cases, preds, contract_level="core")
    cases_path = tmp_path / "cases.jsonl"
    report_path = tmp_path / "report.json"
    audit_path = tmp_path / "audit.json"
    save_jsonl(cases, cases_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    benchmark_cli(
        [
            "audit",
            "--report",
            str(report_path),
            "--cases",
            str(cases_path),
            "--out",
            str(audit_path),
        ]
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["summary"]["n_criteria"] == len(criteria_catalog())
    assert audit["summary"]["has_case_truth_audit"]
    assert audit["truth_field_availability"]["j_call"]["n_present"] == len(cases)
