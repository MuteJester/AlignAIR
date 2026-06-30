import json

from alignair.benchmark import (
    audit_criteria_report,
    grade_metric_value,
    metric_higher_is_better,
    metric_quality,
    metric_registry,
    metric_spec,
    metric_spec_catalog,
    metric_thresholds,
    criteria_catalog,
)
from alignair.benchmark.cli import main as benchmark_cli


def test_metric_registry_covers_all_catalog_metric_keys():
    catalog_keys = {
        key
        for criterion in criteria_catalog()
        for key in criterion.get("metric_keys", ())
    }
    registry = metric_registry()

    assert catalog_keys
    assert catalog_keys <= set(registry)
    assert len(metric_spec_catalog()) == len(registry)


def test_metric_registry_preserves_current_direction_and_threshold_rules():
    call = metric_spec("call_top1_in_set")
    ss_mae = metric_spec("ss_mae")
    optional = metric_spec("optional_field_presence")
    runtime = metric_spec("seconds_per_read")

    assert call.higher_is_better is True
    assert call.pass_threshold == 0.99
    assert call.warn_threshold == 0.95
    assert ss_mae.higher_is_better is False
    assert ss_mae.pass_threshold == 0.5
    assert ss_mae.warn_threshold == 2.0
    assert optional.pass_threshold == 0.75
    assert optional.warn_threshold == 0.25
    assert runtime.higher_is_better is False
    assert runtime.pass_threshold == 1.0
    assert runtime.warn_threshold == 10.0


def test_registry_grading_matches_assay_report_semantics():
    assert grade_metric_value("call_top1_in_set", 1.0)["grade"] == "pass"
    assert grade_metric_value("call_top1_in_set", 0.96)["grade"] == "warn"
    assert grade_metric_value("call_top1_in_set", 0.5)["grade"] == "fail"

    assert grade_metric_value("ss_mae", 0.0)["grade"] == "pass"
    assert grade_metric_value("ss_mae", 1.0)["grade"] == "warn"
    assert grade_metric_value("ss_mae", 3.0)["grade"] == "fail"

    assert metric_quality("ss_mae", 1.0) == 0.5
    assert metric_higher_is_better("false_positive_alignment_rate") is False
    assert metric_thresholds("reads_per_second") == {
        "higher_is_better": True,
        "pass": 1.0,
        "warn": 0.1,
    }


def test_unknown_metric_gets_default_metadata_without_entering_catalog():
    spec = metric_spec("custom_debug_metric")

    assert spec.key == "custom_debug_metric"
    assert spec.criterion_names == ()
    assert spec.categories == ()
    assert spec.higher_is_better is True


def test_criteria_audit_uses_registry_metadata_for_catalog_metrics():
    scores = {"global": {"call_top1_in_set": 1.0}, "genes": {}}
    audit = audit_criteria_report(scores)
    by_name = {row["name"]: row for row in audit["criteria"]}
    allele = by_name["allele_top1_call"]
    spec = allele["metric_specs"][0]

    assert audit["summary"]["n_registered_metric_keys"] == len(metric_registry())
    assert audit["summary"]["n_unregistered_catalog_metric_keys"] == 0
    assert audit["summary"]["n_observed_metric_keys_without_registry"] == 0
    assert allele["registered_metric_keys"] == ("call_top1_in_set",)
    assert allele["unregistered_metric_keys"] == ()
    assert spec["key"] == "call_top1_in_set"
    assert spec["registered"] is True
    assert spec["higher_is_better"] is True
    assert spec["pass_threshold"] == 0.99
    assert spec["warn_threshold"] == 0.95
    assert spec["criteria"] == ("allele_top1_call",)


def test_criteria_audit_flags_custom_unregistered_metrics():
    criteria = [
        {
            "category": "unit",
            "name": "custom_debug_criterion",
            "metric_keys": ("custom_debug_metric",),
            "status": "available",
            "importance": "diagnostic",
            "ground_truth_fields": (),
        }
    ]
    scores = {"global": {"custom_debug_metric": 0.4}, "genes": {}}

    audit = audit_criteria_report(scores, criteria=criteria)
    row = audit["criteria"][0]

    assert audit["summary"]["n_catalog_metric_keys"] == 1
    assert audit["summary"]["n_unregistered_catalog_metric_keys"] == 1
    assert audit["summary"]["n_observed_metric_keys_without_registry"] == 1
    assert audit["unregistered_catalog_metric_keys"] == ["custom_debug_metric"]
    assert audit["observed_metric_keys_without_registry"] == ["custom_debug_metric"]
    assert row["registered_metric_keys"] == ()
    assert row["unregistered_metric_keys"] == ("custom_debug_metric",)
    assert row["metric_specs"][0]["registered"] is False


def test_cli_metrics_catalog_is_machine_readable(capsys):
    benchmark_cli(["metrics"])

    payload = json.loads(capsys.readouterr().out)
    catalog = {row["key"]: row for row in payload["metric_registry"]}
    assert "call_top1_in_set" in catalog
    assert catalog["call_top1_in_set"]["higher_is_better"] is True
    assert catalog["ss_mae"]["higher_is_better"] is False


def test_cli_scoring_manifest_is_machine_readable(capsys):
    benchmark_cli(["scoring-manifest"])

    payload = json.loads(capsys.readouterr().out)
    catalog = {row["name"]: row for row in payload["scoring_manifest"]}
    assert "gene" in catalog
    assert "orientation" in catalog
    assert catalog["gene"]["scope"] == "gene"
    assert "call_top1_in_set" in catalog["gene"]["metric_keys"]
