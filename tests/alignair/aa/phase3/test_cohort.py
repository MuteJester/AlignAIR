import pytest
import hashlib

from alignair.aa.phase3.cohort import (
    validate_row, validate_cohort, sha256_seq, LabelRow, LABEL_FIELDS,
)

pytestmark = pytest.mark.component

SEQ = "EVQLVESGGGLVQPGGSLRLSCAAS"
VID = hashlib.sha256(SEQ.encode()).hexdigest()
CROSS = {"VH subgroup III": ("IGHV3", "", "Scaviner 1999 IMGT")}


def _row(**kw):
    d = dict(name="examplemab", arm_id="", exact_vh_id=VID, lineage_id="examplemab",
             lineage_representative="true", chain_class="in_scope", seq_source="therasabdab",
             seq_source_version="2024-12-20", retrieval_date="2026-07-21",
             raw_file_sha256="a" * 64, true_v_family="IGHV3", true_v_gene="", true_v_allele="",
             label_resolution="subgroup_only", in_catalog="true", tier="gold",
             label_source="VH subgroup III", label_citation="US1234567 Example 1",
             alias_source="Scaviner 1999 IMGT", evidence_locator="col5 Ex1",
             humanization_status="humanized", true_j="", exclusion_reason="")
    d.update(kw)
    return d


def test_valid_gold_row_passes_row_check():
    assert validate_row(_row()) == []


def test_row_requires_citation_and_evidence_and_family():
    assert any("citation" in v for v in validate_row(_row(label_citation="")))
    assert any("evidence" in v for v in validate_row(_row(evidence_locator="")))
    assert any("family" in v for v in validate_row(_row(true_v_family="")))


def test_row_rejects_bad_enums_hash_date_boolean():
    assert any("boolean" in v for v in validate_row(_row(lineage_representative="yes")))
    assert any("tier" in v for v in validate_row(_row(tier="silver")))
    assert any("chain_class" in v for v in validate_row(_row(chain_class="weird")))
    assert any("hash" in v for v in validate_row(_row(raw_file_sha256="short")))
    assert any("date" in v for v in validate_row(_row(retrieval_date="2026/07/21")))


def test_cohort_checks_hash_matches_sequence():
    rows = [LabelRow(**_row())]
    assert validate_cohort(rows, {VID: SEQ}, CROSS) == []
    assert any("exact_vh_id" in v for v in validate_cohort(rows, {VID: "DIFFERENT"}, CROSS))


def test_cohort_requires_exactly_one_deterministic_representative():
    a = _row(name="m", lineage_id="L", exact_vh_id="b" * 64, lineage_representative="true")
    b = _row(name="m2", lineage_id="L", exact_vh_id="a" * 64, lineage_representative="false")
    rows = [LabelRow(**a), LabelRow(**b)]
    hy = {"b" * 64: "AAAA", "a" * 64: "CCCC"}
    # representative must be the lexicographically smallest exact_vh_id ("a..") — here it's on the wrong row
    assert any("representative" in v for v in validate_cohort(rows, hy, CROSS))


def test_cohort_rejects_conflicting_labels_for_same_vh():
    a = _row(name="x", exact_vh_id=VID, true_v_family="IGHV3")
    b = _row(name="y", exact_vh_id=VID, true_v_family="IGHV1", lineage_id="y")
    assert any("conflict" in v for v in validate_cohort([LabelRow(**a), LabelRow(**b)], {VID: SEQ}, CROSS))


def test_cohort_gold_requires_alias_in_crosswalk():
    r = _row(label_source="Unlisted framework X")
    assert any("crosswalk" in v for v in validate_cohort([LabelRow(**r)], {VID: SEQ}, CROSS))


def test_cohort_concordance_must_be_marked_alignment_derived():
    r = _row(tier="concordance", true_v_gene="IGHV3-23", label_resolution="exact_gene",
             label_source="hand pick", alias_source="")
    assert any("alignment-derived" in v for v in validate_cohort([LabelRow(**r)], {VID: SEQ}, CROSS))
