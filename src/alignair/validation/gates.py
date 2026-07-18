"""Version-controlled release gates.

Two things live here, and both are code (so changing them is a reviewed change, never a silent
bypass):

* ``SCIENTIFIC_THRESHOLDS`` — the per-task floors a production model must clear on a fixed, seeded,
  low-corruption validation set. Enforced by
  ``tests/alignair/validation/test_release_gates.py::test_igh_model_meets_scientific_gates``.
* ``CLAIM_TESTS`` — the map from each product / scientific CLAIM the model card makes to the named
  automated test that proves it. A test asserts every entry points at a real test, so a claim can never
  drift away from its evidence.
"""
from __future__ import annotations

# Per-task floors on a fixed seeded validation batch (moderate corruption). Conservative — a healthy
# production model clears them comfortably; a regression that drops below is a release blocker.
SCIENTIFIC_THRESHOLDS = {
    "v_allele_top1": 0.75,     # V allele top-1-in-set
    "j_allele_top1": 0.85,     # J allele top-1-in-set
    "d_allele_top1": 0.25,     # D is intrinsically hard (short, mutated)
    "productive_acc": 0.75,
    "orientation_acc": 0.90,
}

# Each product/scientific claim -> the test (pytest nodeid) that enforces it.
CLAIM_TESTS = {
    "segment coordinates are bounded and ordered":
        "tests/alignair/predict/test_segment.py::test_property_bounds_and_ordering",
    "genotype-constrained calls stay within the allowed set":
        "tests/alignair/predict/test_threshold.py::test_allowed_set_restricts_call_even_when_disallowed_has_max_prob",
    "cross-locus calls are impossible by construction":
        "tests/alignair/predict/test_multichain_locus.py::test_locus_allowed_restricts_each_read_to_its_locus",
    "novel alleles are rejected (fixed-reference contract)":
        "tests/alignair/test_fixed_reference_contract.py::test_novel_allele_is_rejected_not_dropped",
    "AIRR output passes official validation":
        "tests/alignair/golden/test_golden_airr.py::test_golden_airr_passes_official_validation",
    "productive is a derived fact, not the neural prediction":
        "tests/alignair/predict/airr/test_semantics.py::test_airr_productive_is_derived_from_frame_and_stop",
    "no advertised output column is undocumented":
        "tests/alignair/io/test_field_map.py::test_every_emitted_column_is_documented",
    "over-length reads are cropped and flagged, never silently truncated":
        "tests/alignair/io/test_sequence_reader.py::test_apply_input_policy_crops_over_length_and_flags",
    "AIRR assembly failures are tagged, not swallowed":
        "tests/alignair/predict/airr/test_airr.py::test_build_airr_expected_exception_is_tagged_not_swallowed",
    "orientation transforms are involutions":
        "tests/alignair/property/test_invariants.py::test_canonicalize_is_an_involution",
    "the CLI and Python API agree on the same input":
        "tests/alignair/cli/test_cli_api_parity.py::test_cli_matches_python_api",
    "training aborts on a non-finite loss":
        "tests/alignair/train/test_guards.py::test_non_finite_loss_aborts",
    "concurrent cache access is mutually exclusive":
        "tests/alignair/registry/test_lock.py::test_excl_lock_serializes_concurrent_holders",
}
