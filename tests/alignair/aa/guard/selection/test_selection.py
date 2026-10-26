from types import SimpleNamespace

import pytest

from alignair.aa.guard.decision import CandidateSpec, candidate_grid
from alignair.aa.guard.selection import (
    SELECTION_GATES, group_counts, passes_all_gates, select_thresholds,
)
from alignair.aa.guard.thresholds import clopper_pearson_upper

pytestmark = pytest.mark.component

SPEC = CandidateSpec.of("C2")


def _rec(group, *, accept, unit_id, parent=None, rid="r"):
    """A minimal blind RecordScore-like. accept=True means the decision graph would accept (supported)."""
    comp = (5.0, 0, 0, 0, 0) if accept else (0, 5.0, 0, 0, 0)
    typ = (5.0, 0, 0, 0, 0, 0, 0, 0) if accept else (0, 5.0, 0, 0, 0, 0, 0, 0)
    hmm = ({"has_v": True, "has_j": True, "order_ok": True, "combined_coverage": 0.95,
            "best_v_group": "human_igh", "best_alt_score": 0.0, "human_igh_score": 30.0} if accept else
           {"has_v": True, "has_j": True, "order_ok": True, "combined_coverage": 0.95,
            "best_v_group": "human_igl", "best_alt_score": 30.0, "human_igh_score": 0.0})
    return SimpleNamespace(within_length_contract=True, hmm_evidence=hmm, completeness_logits=comp,
                           domain_type_logits=typ, applicability_logit=(5.0 if accept else -5.0), energy=-10.0,
                           group=group, record_id_sha256=rid,
                           unit={"lineage": unit_id, "parent_lineage": parent or unit_id})


def test_far_and_frr_directions_and_internal_counts():
    gp = candidate_grid("C2")[0]
    # a negative group where every unit is (wrongly) accepted -> all units are FAR failures
    neg = [_rec("igk_igl", accept=True, unit_id=f"L{i}") for i in range(10)]
    tbl = group_counts(neg, SPEC, gp)
    assert tbl["igk_igl"].k == 10 and tbl["igk_igl"].n == 10 and not tbl["igk_igl"].passed
    # a positive group where every unit is (wrongly) rejected -> all units are FRR failures
    pos = [_rec("positive_real", accept=False, unit_id=f"L{i}") for i in range(10)]
    tbl2 = group_counts(pos, SPEC, gp)
    assert tbl2["positive_real"].k == 10 and not tbl2["positive_real"].passed


def test_structural_counts_aggregate_by_parent_lineage():
    gp = candidate_grid("C2")[0]
    # 100 structural records but only 4 distinct parent lineages -> n == 4 units, not 100
    recs = [_rec("structural", accept=False, unit_id=f"rec{i}", parent=f"P{i % 4}") for i in range(100)]
    tbl = group_counts(recs, SPEC, gp)
    assert SELECTION_GATES["structural"]["unit"] == "parent_lineage"
    assert tbl["structural"].n == 4 and tbl["structural"].k == 0        # zero false-accepts across 4 lineages


def test_no_admissible_threshold_returns_none_stop():
    gp_scores = [_rec("igk_igl", accept=True, unit_id=f"L{i}") for i in range(80)]   # always false-accepts
    assert select_thresholds(SPEC, gp_scores, ["igk_igl"]) is None      # no grid point can fix a always-accept


def test_one_group_failure_blocks_admissibility_despite_others_passing():
    # igk_igl: 80 clean negatives (rejected) -> passes at a strict point
    clean = [_rec("igk_igl", accept=False, unit_id=f"K{i}") for i in range(80)]
    # igsf: a single false-accept in only 40 units -> upper bound over the 10% ceiling -> fails
    bad = [_rec("igsf", accept=(i == 0), unit_id=f"S{i}") for i in range(40)]
    table = group_counts(clean + bad, SPEC, candidate_grid("C2")[0])
    assert table["igk_igl"].passed and not table["igsf"].passed
    # Pooling would hide the failing group, even under the stricter FAR ceiling.
    pooled_upper = clopper_pearson_upper(sum(row.k for row in table.values()),
                                        sum(row.n for row in table.values()))
    assert pooled_upper <= min(row.ceiling for row in table.values())
    assert select_thresholds(SPEC, clean + bad, ["igk_igl", "igsf"]) is None
    assert select_thresholds(SPEC, clean, ["igk_igl"]) is not None      # igk_igl alone is admissible


@pytest.mark.parametrize("failing,clean", [("positive_real", "positive_sim"), ("positive_sim", "positive_real")])
def test_real_and_simulated_positive_failures_cannot_compensate_for_each_other(failing, clean):
    scores = ([_rec(failing, accept=(index >= 2), unit_id=f"bad-{index}") for index in range(40)]
              + [_rec(clean, accept=True, unit_id=f"clean-{index}") for index in range(60)])
    table = group_counts(scores, SPEC, candidate_grid("C2")[0])
    assert (table[failing].k, table[failing].n, table[failing].passed) == (2, 40, False)
    assert (table[clean].k, table[clean].n, table[clean].passed) == (0, 60, True)
    assert clopper_pearson_upper(2, 100) <= table[failing].ceiling  # pooling would pass
    assert select_thresholds(SPEC, scores, [failing, clean]) is None
    assert select_thresholds(SPEC, scores, [clean]) is not None


def test_passing_multi_group_selection_first_admissible():
    scores = ([_rec("igk_igl", accept=False, unit_id=f"K{i}") for i in range(80)]     # 5%-gate, zero error
              + [_rec("igsf", accept=False, unit_id=f"S{i}") for i in range(40)]       # 10%-gate, zero error
              + [_rec("positive_real", accept=True, unit_id=f"R{i}") for i in range(60)]   # FRR zero error
              + [_rec("positive_sim", accept=True, unit_id=f"M{i}") for i in range(60)])
    req = ["igk_igl", "igsf", "positive_real", "positive_sim"]
    sel = select_thresholds(SPEC, scores, req)
    assert sel is not None and sel.gates_pass
    assert passes_all_gates({g: SimpleNamespace(**sel.per_group[g]) for g in sel.per_group}, req)
    for g in req:
        assert sel.per_group[g]["passed"] and sel.per_group[g]["k"] == 0
