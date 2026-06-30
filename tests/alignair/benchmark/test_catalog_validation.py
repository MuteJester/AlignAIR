import json

from alignair.benchmark import criteria_catalog, scenario_axes_catalog, validate_catalogs
from alignair.benchmark.cli import main as benchmark_cli


def test_catalog_validation_accepts_current_assay_catalogs():
    validation = validate_catalogs()
    axes = {axis["name"]: axis for axis in scenario_axes_catalog()}

    assert validation["valid"] is True
    assert validation["warnings"] == ()
    assert validation["summary"]["n_criteria"] == len(criteria_catalog())
    assert validation["summary"]["n_scenario_axes"] == len(scenario_axes_catalog())
    assert validation["summary"]["n_unmapped_contexts"] == 0
    assert validation["summary"]["n_metric_keys_without_registry"] == 0
    assert "contaminant" in axes["input_validity"]["values"]


def test_catalog_validation_flags_structural_catalog_errors():
    criteria = [
        {
            "category": "",
            "name": "duplicate",
            "metric_keys": ("unknown_metric", "unknown_metric"),
            "description": "",
            "contexts": ("unmapped_context",),
            "importance": "unsupported",
            "status": "experimental",
            "interpretation": "",
        },
        {
            "category": "unit",
            "name": "duplicate",
            "metric_keys": (),
            "description": "Missing metrics.",
            "contexts": (),
            "importance": "core",
            "status": "available",
            "interpretation": "Invalid on purpose.",
        },
    ]
    scenario_axes = [
        {"name": "axis", "values": ("x", "x"), "description": "", "why_it_matters": ""},
        {"name": "axis", "values": (), "description": "Duplicate axis.", "why_it_matters": "Invalid."},
    ]

    validation = validate_catalogs(criteria=criteria, scenario_axes=scenario_axes)

    assert validation["valid"] is False
    assert "duplicate_criterion_names" in validation["problems"]
    assert "duplicate_scenario_axis_names" in validation["problems"]
    assert "invalid_criteria" in validation["problems"]
    assert "invalid_scenario_axes" in validation["problems"]
    assert "criteria_metric_keys_without_registry" in validation["problems"]
    assert validation["warnings"] == ("criteria_contexts_without_scenario_axis",)
    assert validation["duplicate_criterion_names"] == ("duplicate",)
    assert validation["metric_keys_without_registry"] == ("unknown_metric",)


def test_cli_validate_catalogs_writes_machine_readable_report(tmp_path):
    out = tmp_path / "catalog_validation.json"

    benchmark_cli(["validate-catalogs", "--out", str(out), "--fail-on-warning"])

    validation = json.loads(out.read_text(encoding="utf-8"))
    assert validation["valid"] is True
    assert validation["warnings"] == []
    assert validation["summary"]["n_unmapped_contexts"] == 0
