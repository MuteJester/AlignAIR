"""Task 2: the split-aware feasibility calculator powers the LOCKED TEST (not the total cohort),
deduplicates by independent lineage, excludes gamma/delta from the TCR floor, requires both loci in
combined groups, and counts structural negatives by PARENT lineage."""
from __future__ import annotations
import pytest

from alignair.aa.product.applicability.cohort import CohortEntry
from alignair.aa.product.applicability.feasibility import (
    FLOOR_5, FLOOR_10, feasibility_report, min_n_zero_error, min_n_for_upper_bound)

pytestmark = pytest.mark.component


def _entries(category, group, state, n, *, lineage_prefix=None, shared_lineage=None):
    """n entries; distinct lineage keys unless `shared_lineage` collapses them (dedup test)."""
    pre = lineage_prefix or category
    out = []
    for i in range(n):
        lk = shared_lineage if shared_lineage is not None else f"{pre}_{i}"
        out.append(CohortEntry(id=f"{category}_{i}", category=category, group=group,
                               expected_state=state, lineage_key=lk))
    return out


def _powered_cohort():
    """A cohort that clears every gated group's locked-test floor."""
    e = []
    e += _entries("igk", "igk_igl", "unknown_ood", 60)
    e += _entries("igl", "igk_igl", "unknown_ood", 60)          # combined 120 >= 118
    e += _entries("tra", "tcr", "unknown_ood", 60)
    e += _entries("trb", "tcr", "unknown_ood", 60)               # 120 >= 118 (gamma/delta excluded)
    e += _entries("trg", "tcr", "unknown_ood", 5)               # descriptive, present
    e += _entries("murine_vh", "vhh_nonhuman", "unknown_ood", 30)
    e += _entries("vhh", "vhh_nonhuman", "unknown_ood", 30)      # 60 >= 58
    e += _entries("igsf_vset", "igsf", "unknown_ood", 60)        # >= 58
    e += _entries("reversed_vh", "structural", "insufficient_evidence", 60)   # 60 parent lineages >= 58
    e += _entries("pos_igh_repertoire", "positive_real", "supported", 120)    # >= 116
    e += _entries("pos_igh_simulator", "positive_sim", "supported", 120)      # >= 116
    return e


def test_floors_match_the_spec():
    assert FLOOR_5 == 59 and min_n_zero_error(0.05) == 59
    assert FLOOR_10 == 29 and min_n_zero_error(0.10) == 29


def test_min_n_rises_with_observed_errors():
    # zero errors -> 59; one observed error requires strictly more to still bound <= 5%
    assert min_n_for_upper_bound(0.05, 0) == 59
    assert min_n_for_upper_bound(0.05, 1) > 59


def test_powered_cohort_is_GO():
    rep = feasibility_report(_powered_cohort())
    assert rep["decision"] == "GO", rep["stops"]
    assert rep["groups"]["tcr"]["descriptive_present"] == 5          # gamma/delta present, not counted
    assert rep["groups"]["tcr"]["counts_toward_floor"] == 120        # only TRA/TRB


def test_missing_locus_forces_STOP():
    cohort = [e for e in _powered_cohort() if e.category != "igl"]   # drop IGL entirely
    rep = feasibility_report(cohort)
    assert rep["decision"] == "STOP"
    assert rep["groups"]["igk_igl"]["both_loci_present"] is False
    assert "igl" in rep["groups"]["igk_igl"]["missing_loci"]


def test_gamma_delta_alone_cannot_power_tcr():
    # replace TRA/TRB with only a large gamma/delta set -> floor count 0 -> STOP
    cohort = [e for e in _powered_cohort() if e.category not in ("tra", "trb")]
    cohort += _entries("trg", "tcr", "unknown_ood", 200)
    rep = feasibility_report(cohort)
    assert rep["groups"]["tcr"]["counts_toward_floor"] == 0          # descriptive gamma/delta never count
    assert rep["groups"]["tcr"]["descriptive_present"] > 0           # they ARE present, just not gated
    assert rep["decision"] == "STOP"


def test_structural_dedup_counts_parent_lineages_not_transforms():
    # 30 parents x 6 transforms = 180 rows but only 30 independent parent lineages -> under floor 58
    cohort = [e for e in _powered_cohort() if e.category != "reversed_vh"]
    for t in ("reversed_vh", "cdr3_only", "v_only", "j_only", "leader_vh", "vh_constant"):
        # every transform shares its parent's lineage key -> dedup collapses to 30
        for p in range(30):
            cohort.append(CohortEntry(id=f"{t}_{p}", category=t, group="structural",
                                      expected_state="insufficient_evidence", lineage_key=f"parent_{p}"))
    rep = feasibility_report(cohort)
    assert rep["groups"]["structural"]["effective_after_dedup"] == 30   # not 180
    assert rep["decision"] == "STOP"
    assert any("structural" in s for s in rep["stops"])


def test_underpowered_positive_forces_STOP():
    cohort = [e for e in _powered_cohort() if e.category != "pos_igh_repertoire"]
    cohort += _entries("pos_igh_repertoire", "positive_real", "supported", 100)   # < 116
    rep = feasibility_report(cohort)
    assert rep["decision"] == "STOP"
    assert rep["groups"]["positive_real"]["powered"] is False
