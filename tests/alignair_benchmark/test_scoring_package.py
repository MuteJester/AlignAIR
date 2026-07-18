from alignair_benchmark.evaluation import metrics, scoring
from alignair_benchmark.evaluation.scoring.manifest import (
    GENE_MANIFEST,
    scoring_manifest_catalog,
    validate_scoring_manifest,
)
from alignair_benchmark.evaluation.scoring.runtime_audit import audit_scoring_runtime
from alignair_benchmark.evaluation.scoring.registry import (
    GLOBAL_SCORING_COMPONENTS,
    validate_global_scoring_registry,
)


def test_metrics_facade_reexports_scoring_api() -> None:
    assert metrics.score_one_case is scoring.score_one_case
    assert metrics.score_cases is scoring.score_cases
    assert metrics.compact_summary is scoring.compact_summary


def test_scoring_package_exports_public_api_only() -> None:
    assert set(scoring.__all__) == {
        "audit_scoring_runtime",
        "compact_summary",
        "score_cases",
        "score_one_case",
        "scoring_manifest_catalog",
        "validate_scoring_manifest",
    }


def test_global_scoring_registry_is_valid() -> None:
    assert validate_global_scoring_registry() == []
    names = {component.name for component in GLOBAL_SCORING_COMPONENTS}
    assert {
        "orientation",
        "airr_contract",
        "segment_order",
        "junction",
        "metadata",
        "performance",
        "region_labels",
        "state_labels",
    } <= names


def test_scoring_manifest_is_valid_and_covers_components() -> None:
    validation = validate_scoring_manifest()
    assert validation["valid"] is True
    assert validation["metric_keys_without_registry"] == ()

    manifest_names = {row["name"] for row in scoring_manifest_catalog(include_metric_specs=False)}
    component_names = {component.name for component in GLOBAL_SCORING_COMPONENTS}
    assert component_names | {GENE_MANIFEST.name} == manifest_names


def test_scoring_manifest_catalog_includes_metric_semantics() -> None:
    catalog = scoring_manifest_catalog()
    by_name = {row["name"]: row for row in catalog}

    orientation = by_name["orientation"]
    assert orientation["scope"] == "global"
    assert orientation["metrics"][0]["key"] == "orientation_acc"
    assert orientation["metrics"][0]["registered"] is True
    assert orientation["metrics"][0]["direction"] == "higher_is_better"


def test_scoring_runtime_audit_separates_declared_and_observed_metrics() -> None:
    scores = {
        "global": {"orientation_acc": 1.0, "custom_debug_metric": 0.5},
        "genes": {"v": {"call_top1_in_set": 1.0, "ss_mae": 0.0}},
    }
    audit = audit_scoring_runtime(scores)

    assert audit["summary"]["n_components"] >= 2
    assert audit["summary"]["n_observed_metric_keys_without_manifest"] == 1
    assert audit["observed_metric_keys_without_manifest"] == ("custom_debug_metric",)
    by_name = {row["name"]: row for row in audit["components"]}
    assert by_name["orientation"]["status"] == "fully_exercised"
    assert by_name["gene"]["status"] == "partially_exercised"
    assert "call_top1_in_set" in by_name["gene"]["observed_metric_keys"]
