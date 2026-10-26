"""Criteria, scenario-axis, and comparison-policy catalogs."""
import pytest
from alignair_benchmark import (
    comparison_policy_catalog,
    criteria_catalog,
    scenario_axes_catalog,
    validate_comparison_policy_catalog,
)

pytestmark = pytest.mark.component
pytest.importorskip("GenAIRR")


def test_comparison_policy_catalog_validates_against_metric_registry():
    validation = validate_comparison_policy_catalog()

    assert validation["valid"] is True
    assert validation["summary"]["n_policies"] == len(comparison_policy_catalog())
    assert validation["summary"]["n_invalid_policy_metrics"] == 0
    assert any(
        row["metric"] == "genes.v.call_top1_in_set" and row["registered"]
        for row in validation["policy_metrics"]
    )

    broken = validate_comparison_policy_catalog(
        {
            "bad_policy": {
                "primary_metrics": ("genes.v.not_a_metric", "bad.path.shape"),
                "guardrail_metrics": (),
            }
        }
    )
    assert broken["valid"] is False
    assert broken["summary"]["problem_counts"]["metric_key_not_in_registry"] == 2
    assert broken["summary"]["problem_counts"]["malformed_metric_path"] == 1


def test_criteria_catalog_names_core_assay_dimensions():
    catalog = criteria_catalog()
    names = {c["name"] for c in catalog}
    assert {
        "airr_schema_completeness",
        "coordinate_convention_compliance",
        "locus_and_chain_support",
        "orientation_detection",
        "prediction_completeness",
        "in_sequence_boundaries",
        "junction_sequence_and_translation",
        "np_and_p_nucleotide_recovery",
        "germline_boundaries",
        "alignment_path_and_cigar",
        "set_valued_allele_call",
        "sibling_allele_resolution",
        "genotype_mask_compliance",
        "constant_region_call",
        "d_orientation_and_inversion",
        "regional_shm_profile",
        "segment_identity_and_support",
        "allele_uncertainty_calibration",
        "fragment_observability",
        "read_layout_and_end_loss",
        "contaminant_and_out_of_scope_handling",
        "receptor_revision_cases",
        "productive_status",
    }.issubset(names)
    categories = {c["category"] for c in catalog}
    assert {
        "airr_output_contract",
        "alignment_quality",
        "allele_calling",
        "calibration",
        "chain_locus",
        "efficiency",
        "robustness",
    }.issubset(categories)
    statuses = {c["status"] for c in catalog}
    assert {"available", "partial", "planned"}.issubset(statuses)
    assert all("ground_truth_fields" in c for c in catalog)


def test_scenario_axes_catalog_names_stress_dimensions():
    axes = {a["name"]: a for a in scenario_axes_catalog()}
    assert {
        "difficulty_stratum",
        "length",
        "locus_chain",
        "mutation_burden",
        "indel_burden",
        "allele_ambiguity",
        "d_orientation",
        "genotype_size",
        "read_layout",
        "segment_presence",
        "junction_biology",
        "input_validity",
    }.issubset(axes)
    assert "fragment_50" in axes["difficulty_stratum"]["values"]
    assert "contaminant" in axes["input_validity"]["values"]
