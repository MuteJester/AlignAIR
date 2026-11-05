"""The release claims of AlignAIR's model card, each mapped to the named automated test that proves it (a pytest
node id). ``tests.support.policy`` marks those tests ``claim_evidence``, and ``--enforce-claims`` fails a run in which
one of them was not selected or did not pass; ``tests/quality/test_claims.py`` checks that every entry names a real
test, so a claim can never drift away from its evidence."""

CLAIM_TESTS = {
    "segment coordinates are bounded and ordered":
        "tests/alignair/predict/test_segment.py::test_property_bounds_and_ordering",
    "genotype-constrained calls stay within the allowed set":
        "tests/alignair/predict/test_threshold.py::test_allowed_set_restricts_call_even_when_disallowed_has_max_prob",
    "novel alleles are rejected (fixed-reference contract)":
        "tests/acceptance/test_fixed_reference_contract.py::test_novel_allele_is_rejected_not_dropped",
    "AIRR output passes official validation":
        "tests/acceptance/test_airr_validation.py::test_golden_airr_passes_official_validation",
    "productivity is derived from the predicted anchors' junction (in frame, no stop codon)":
        "tests/alignair/predict/test_junction.py::test_an_in_frame_junction_without_a_stop_is_productive",
    "over-length reads are cropped and flagged, never silently truncated":
        "tests/alignair/data/test_input_policies.py::test_prediction_crops_over_length_and_flags_exact_short_and_long_reads",
    "AIRR assembly failures are tagged, not swallowed":
        "tests/alignair/predict/airr/test_airr.py::test_build_airr_expected_exception_is_tagged_not_swallowed",
    "orientation transforms are involutions":
        "tests/robustness/test_invariants.py::test_canonicalize_is_an_involution",
    "the CLI and Python API agree on the same input":
        "tests/alignair/cli/test_cli_api_parity.py::test_cli_matches_python_api",
    "training aborts on a non-finite loss":
        "tests/alignair/train/test_guards.py::test_train_step_rejects_non_finite_loss_or_gradient_before_optimizer",
    "concurrent cache access is mutually exclusive":
        "tests/alignair/registry/test_lock.py::test_excl_lock_serializes_concurrent_holders",
}
