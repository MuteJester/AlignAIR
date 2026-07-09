import json

import pytest

from alignair_benchmark import (
    BENCHMARK_CASE_JSONL,
    BENCHMARK_MANIFEST,
    BENCHMARK_READINESS_REPORT,
    BENCHMARK_REPORT,
    BENCHMARK_SUITE_MANIFEST,
    CURRENT_SCHEMA_VERSION,
    MODEL_COMPARISON_REPORT,
    BenchmarkCase,
    GeneTruth,
    assess_benchmark_readiness,
    artifact_contract_catalog,
    build_benchmark_manifest,
    build_benchmark_report,
    build_model_comparison_report,
    case_to_prediction,
    metric_registry,
    validate_artifact,
    validate_benchmark_report_contract,
)
from alignair_benchmark.cli import main as benchmark_cli


def _case() -> BenchmarkCase:
    genes = {
        "v": GeneTruth(("IGHV1-1*01",), "IGHV1-1*01", 0, 80, 0, 80),
        "d": GeneTruth(),
        "j": GeneTruth(("IGHJ4*01",), "IGHJ4*01", 80, 100, 0, 20),
    }
    return BenchmarkCase(
        case_id="artifact-1",
        stratum="artifact",
        sequence="A" * 100,
        canonical_sequence="A" * 100,
        orientation_id=0,
        genes=genes,
        presented_genes=genes,
        scalars={"productive": 1.0},
        record={"productive": True},
    )


def test_artifact_contract_catalog_names_current_artifacts():
    catalog = {row["kind"]: row for row in artifact_contract_catalog()}

    assert {
        BENCHMARK_CASE_JSONL,
        BENCHMARK_REPORT,
        BENCHMARK_MANIFEST,
        BENCHMARK_READINESS_REPORT,
        BENCHMARK_SUITE_MANIFEST,
        MODEL_COMPARISON_REPORT,
    } <= set(catalog)
    assert catalog[BENCHMARK_CASE_JSONL]["schema_version"] == CURRENT_SCHEMA_VERSION
    assert catalog[BENCHMARK_REPORT]["schema_name"].endswith(f".v{CURRENT_SCHEMA_VERSION}")
    assert "catalog_validation" in catalog[BENCHMARK_REPORT]["optional_fields"]
    assert "metric_registry" in catalog[BENCHMARK_REPORT]["optional_fields"]
    assert "scoring_manifest" in catalog[BENCHMARK_REPORT]["optional_fields"]
    assert "scoring_audit" in catalog[BENCHMARK_REPORT]["optional_fields"]
    assert "case_id" in catalog[BENCHMARK_CASE_JSONL]["required_fields"]
    assert "components" in catalog[BENCHMARK_SUITE_MANIFEST]["required_fields"]
    assert "suite_readiness" in catalog[BENCHMARK_SUITE_MANIFEST]["required_fields"]


def test_case_jsonl_contract_validates_current_row_shape():
    row = _case().to_dict()
    validation = validate_artifact(row, BENCHMARK_CASE_JSONL)

    assert validation["valid"] is True
    assert validation["version_present"] is False
    assert validation["missing_required_fields"] == ()

    broken = dict(row)
    broken.pop("sequence")
    validation = validate_artifact(broken, BENCHMARK_CASE_JSONL)
    assert validation["valid"] is False
    assert validation["problems"] == ("missing_required_fields",)
    assert validation["missing_required_fields"] == ("sequence",)


def test_benchmark_report_embeds_and_validates_artifact_metadata():
    case = _case()
    report = build_benchmark_report([case], [case_to_prediction(case)], contract_level="minimal")

    assert report["artifact"]["kind"] == BENCHMARK_REPORT
    assert report["artifact"]["schema_version"] == CURRENT_SCHEMA_VERSION
    assert report["catalog_validation"]["valid"] is True
    assert report["catalog_validation"]["warnings"] == ()
    assert {row["key"] for row in report["metric_registry"]} == set(metric_registry())
    assert "scoring_manifest" in report
    assert report["scoring_audit"]["summary"]["has_case_truth_audit"] is True
    assert report["scoring_audit"]["summary"]["has_prediction_field_audit"] is True
    assert report["scoring_audit"]["summary"]["n_observed_metric_keys_without_manifest"] == 0
    assert validate_artifact(report, BENCHMARK_REPORT)["valid"] is True
    report_validation = validate_benchmark_report_contract(report, require_current_version=True)
    assert report_validation["valid"] is True
    assert report_validation["warnings"] == ()
    assert report_validation["criteria"]["unregistered_metric_keys"] == ()
    assert report_validation["observed"]["metric_keys_without_registry"] == ()
    assert report_validation["scoring_manifest"]["present"] is True
    assert report_validation["scoring_manifest"]["metric_keys_without_registry"] == ()
    assert report_validation["scoring_audit_count_mismatches"] == ()

    incompatible = dict(report)
    incompatible["artifact"] = dict(report["artifact"], schema_version="99.0")
    validation = validate_artifact(incompatible, BENCHMARK_REPORT, require_current_version=True)
    assert validation["valid"] is False
    assert validation["problems"] == ("incompatible_schema_version",)

    wrong_kind = dict(report)
    wrong_kind["artifact"] = dict(report["artifact"], kind=BENCHMARK_MANIFEST)
    validation = validate_artifact(wrong_kind, BENCHMARK_REPORT)
    assert validation["valid"] is False
    assert validation["problems"] == ("artifact_kind_mismatch",)


def test_benchmark_report_contract_validator_flags_registry_gaps():
    case = _case()
    report = build_benchmark_report([case], [case_to_prediction(case)], contract_level="minimal")

    legacy = dict(report)
    legacy.pop("metric_registry")
    validation = validate_benchmark_report_contract(legacy)
    assert validation["valid"] is True
    assert validation["warnings"] == ("missing_metric_registry",)

    broken = dict(report)
    broken["metric_registry"] = [dict(report["metric_registry"][0], key="duplicate")] * 2
    validation = validate_benchmark_report_contract(broken, require_current_version=True)
    assert validation["valid"] is False
    assert "duplicate_metric_registry_keys" in validation["problems"]
    assert "metric_registry_differs_from_current" in validation["problems"]

    broken_audit = dict(report)
    broken_audit["scoring_audit"] = {
        **report["scoring_audit"],
        "summary": {
            **report["scoring_audit"]["summary"],
            "n_observed_metric_keys": 999,
        },
    }
    validation = validate_benchmark_report_contract(broken_audit)
    assert validation["valid"] is False
    assert "scoring_audit_count_mismatch" in validation["problems"]


def test_cli_validate_report_writes_validation_json(tmp_path):
    case = _case()
    report = build_benchmark_report([case], [case_to_prediction(case)], contract_level="minimal")
    report_path = tmp_path / "report.json"
    validation_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    benchmark_cli(
        [
            "validate-report",
            "--report",
            str(report_path),
            "--out",
            str(validation_path),
            "--require-current-version",
            "--require-metric-registry",
        ]
    )

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert validation["warnings"] == []
    assert validation["metric_registry"]["present"] is True


def test_cli_validate_report_exits_nonzero_for_invalid_contract(tmp_path):
    case = _case()
    report = build_benchmark_report([case], [case_to_prediction(case)], contract_level="minimal")
    report["metric_registry"] = [dict(report["metric_registry"][0], key="duplicate")] * 2
    report_path = tmp_path / "broken_report.json"
    validation_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        benchmark_cli(
            [
                "validate-report",
                "--report",
                str(report_path),
                "--out",
                str(validation_path),
                "--require-current-version",
            ]
        )

    assert exc.value.code == 1
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["valid"] is False
    assert "duplicate_metric_registry_keys" in validation["problems"]


def test_benchmark_manifest_embeds_and_validates_artifact_metadata():
    case = _case()
    manifest = build_benchmark_manifest([case], readiness_profile=None)

    assert manifest["manifest_version"] == CURRENT_SCHEMA_VERSION
    assert manifest["artifact"]["kind"] == BENCHMARK_MANIFEST
    assert manifest["artifact"]["schema_version"] == CURRENT_SCHEMA_VERSION
    assert validate_artifact(manifest, BENCHMARK_MANIFEST, require_current_version=True)["valid"] is True


def test_readiness_report_embeds_and_validates_artifact_metadata():
    case = _case()
    report = assess_benchmark_readiness([case], profile="smoke")

    assert report["artifact"]["kind"] == BENCHMARK_READINESS_REPORT
    assert report["artifact"]["schema_version"] == CURRENT_SCHEMA_VERSION
    assert validate_artifact(report, BENCHMARK_READINESS_REPORT, require_current_version=True)["valid"] is True

    broken = dict(report)
    broken.pop("checks")
    validation = validate_artifact(broken, BENCHMARK_READINESS_REPORT)
    assert validation["valid"] is False
    assert validation["missing_required_fields"] == ("checks",)


def test_model_comparison_report_embeds_and_validates_artifact_metadata():
    case = _case()
    pred = case_to_prediction(case)
    report = build_model_comparison_report(
        [case],
        [pred],
        [pred],
        metric_paths=("genes.v.call_top1_in_set",),
        include_strata=False,
    )

    assert report["artifact"]["kind"] == MODEL_COMPARISON_REPORT
    assert report["artifact"]["schema_version"] == CURRENT_SCHEMA_VERSION
    assert validate_artifact(report, MODEL_COMPARISON_REPORT, require_current_version=True)["valid"] is True

    broken = dict(report)
    broken.pop("overall")
    validation = validate_artifact(broken, MODEL_COMPARISON_REPORT)
    assert validation["valid"] is False
    assert validation["missing_required_fields"] == ("overall",)


def test_cli_validate_artifact_accepts_registered_artifacts(tmp_path):
    case = _case()
    report = assess_benchmark_readiness([case], profile="smoke")
    report_path = tmp_path / "readiness.json"
    validation_path = tmp_path / "artifact_validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    benchmark_cli(
        [
            "validate-artifact",
            "--path",
            str(report_path),
            "--kind",
            BENCHMARK_READINESS_REPORT,
            "--out",
            str(validation_path),
            "--require-current-version",
        ]
    )

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert validation["artifact_kind"] == BENCHMARK_READINESS_REPORT
    assert validation["embedded_artifact_kind"] == BENCHMARK_READINESS_REPORT


def test_cli_validate_artifact_exits_nonzero_for_wrong_kind(tmp_path):
    case = _case()
    pred = case_to_prediction(case)
    report = build_model_comparison_report(
        [case],
        [pred],
        [pred],
        metric_paths=("genes.v.call_top1_in_set",),
        include_strata=False,
    )
    report_path = tmp_path / "comparison.json"
    validation_path = tmp_path / "artifact_validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        benchmark_cli(
            [
                "validate-artifact",
                "--path",
                str(report_path),
                "--kind",
                BENCHMARK_READINESS_REPORT,
                "--out",
                str(validation_path),
            ]
        )

    assert exc.value.code == 1
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["valid"] is False
    assert "artifact_kind_mismatch" in validation["problems"]


def test_cli_artifact_contracts_catalog_is_machine_readable(capsys):
    benchmark_cli(["artifact-contracts"])

    payload = json.loads(capsys.readouterr().out)
    catalog = {row["kind"]: row for row in payload["artifact_contracts"]}
    assert BENCHMARK_REPORT in catalog
    assert BENCHMARK_READINESS_REPORT in catalog
    assert MODEL_COMPARISON_REPORT in catalog
    assert catalog[BENCHMARK_REPORT]["schema_version"] == CURRENT_SCHEMA_VERSION
