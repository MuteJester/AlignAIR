"""Task 1: the Gate-3 cohort manifest is provenance-only, covers every category, and reflects the four
settled cohort-design decisions (2026-07-24)."""
from __future__ import annotations
import pytest

import dataclasses

from alignair.aa.product.applicability.cohort import (
    CategorySpec, CohortEntry, DESCRIPTIVE, load_manifest)

pytestmark = pytest.mark.component

_VALID_STATES = {"supported", "insufficient_evidence", "unknown_ood"}


def test_manifest_entries_are_complete_and_provenance_only():
    manifest = load_manifest()
    assert len(manifest) >= 20
    # no sequence field anywhere — provenance only
    field_names = {f.name for f in dataclasses.fields(CategorySpec)}
    assert "sequence" not in field_names
    assert "sequence" not in {f.name for f in dataclasses.fields(CohortEntry)}
    for spec in manifest:
        assert spec.expected_state in _VALID_STATES, spec.key
        for required in ("source", "access", "license", "boundary_source", "group",
                         "lineage_key_scheme"):
            assert getattr(spec, required), f"{spec.key} missing {required}"
        # ground truth must be a declared source, never the Stage-1 rules
        assert "rule" not in spec.boundary_source.lower()


def test_decision_1_gamma_delta_tcr_is_descriptive_not_gated():
    by_key = {s.key: s for s in load_manifest()}
    assert by_key["trg"].gate == DESCRIPTIVE
    assert by_key["trd"].gate == DESCRIPTIVE
    # TRA/TRB carry the combined-TCR floor (not descriptive)
    assert by_key["tra"].gate != DESCRIPTIVE and by_key["trb"].gate != DESCRIPTIVE
    assert by_key["tra"].group == by_key["trg"].group == "tcr"


def test_decision_2_positives_span_repertoire_and_structure():
    by_key = {s.key: s for s in load_manifest()}
    assert "structure" in by_key["pos_igh_structure"].boundary_source.lower()
    assert "sequence_alignment_aa" in by_key["pos_igh_repertoire"].boundary_source \
        or "annotation" in by_key["pos_igh_repertoire"].boundary_source.lower()
    assert by_key["pos_igh_structure"].expected_state == "supported"


def test_decision_3_igsf_allows_cross_species():
    igsf = {s.key: s for s in load_manifest()}["igsf_vset"]
    assert igsf.species == "cross-species"
    assert "cross-species" in igsf.note.lower()


def test_decision_4_vhh_is_sabdab_only_avida_dropped():
    vhh = {s.key: s for s in load_manifest()}["vhh"]
    assert "sabdab" in vhh.source.lower()
    assert "avida" in vhh.note.lower() and "drop" in vhh.note.lower()
    # AVIDa must not appear as a source anywhere
    assert not any("avida" in s.source.lower() for s in load_manifest())


def test_structural_negatives_expect_insufficient_evidence():
    structural = [s for s in load_manifest() if s.gate == "structural"]
    assert {s.key for s in structural} >= {"reversed_vh", "cdr3_only", "v_only", "j_only",
                                           "leader_vh", "vh_constant"}
    for s in structural:
        assert s.expected_state == "insufficient_evidence"
        assert s.expected_reason
        assert "parent lineage" in s.lineage_key_scheme
