import pytest
import hashlib

from alignair.aa.guard.stage1_audit import audit_stage1, combine_allocation, global_dedup

pytestmark = pytest.mark.component


def _oas(study, subj, cat, i):
    lk = f"{study}::{subj}::V::{cat}{i}"
    return {"repository": "oas", "study": study, "donor": subj, "category": cat,
            "group": "igk_igl" if cat in ("igk", "igl") else "positive_real",
            "lineage_key": lk, "aa": f"SEQ{study}{subj}{cat}{i}"}


def _vhh(genus, i):
    return {"category": "vhh", "group": "vhh_nonhuman", "lineage_key": f"vhh-{genus}-{i}",
            "heavy_species": genus, "deposition_date": "2020", "aa": f"VHH{genus}{i}"}


def _igsf(i):
    return {"category": "igsf_vset", "group": "igsf", "lineage_key": f"igsf-{i}", "aa": f"IGSF{i}"}


def _cohort():
    recs = []
    # 12 OAS studies, each 40 igk + 40 igl + 40 heavy from its own donors
    for st in range(12):
        for i in range(40):
            recs.append(_oas(f"St{st}_2020", f"d{st}_{i}", "igk", i))
            recs.append(_oas(f"St{st}_2020", f"d{st}_{i}", "igl", i))
            recs.append(_oas(f"St{st}_2020", f"d{st}_{i}", "pos_igh_repertoire", i))
    recs += [_vhh("lama glama", i) for i in range(60)]
    recs += [_vhh("vicugna pacos", i) for i in range(60)]
    recs += [_igsf(i) for i in range(120)]
    return recs


def test_combine_routes_and_roles_are_disjoint():
    recs = _cohort()
    selection, loco, study_alloc = combine_allocation(recs)
    for g in selection:
        assert not (selection[g] & loco.get(g, set())), f"role overlap in {g}"


def test_audit_passes_on_powered_disjoint_cohort():
    recs = _cohort()
    selection, loco, study_alloc = combine_allocation(recs)
    kept, dropped = global_dedup(recs)
    rep = audit_stage1(records=kept, selection=selection, loco=loco, study_alloc=study_alloc,
                       source_checksums={"sabdab": "abc"},
                       required_groups={"igk_igl", "vhh_nonhuman", "igsf", "positive_real"})
    assert rep.decision == "GO", rep.failures
    assert rep.role_overlap == {"lineage": 0, "donor": 0, "study": 0}
    assert rep.locus_balance_37_37["igk_igl"]
    assert rep.policy_hash_reproduced


def test_audit_flags_donor_straddling_roles():
    recs = _cohort()
    selection, loco, study_alloc = combine_allocation(recs)
    # force a donor's lineage into BOTH roles -> audit must catch donor overlap
    stray = next(iter(loco["igk_igl"]))
    selection["igk_igl"] = set(selection["igk_igl"]) | {stray}
    rep = audit_stage1(records=recs, selection=selection, loco=loco, study_alloc=study_alloc,
                       source_checksums={})
    assert rep.decision.startswith("STOP")
    assert rep.role_overlap["lineage"] >= 1


def test_audit_flags_policy_hash_mismatch():
    recs = _cohort()
    selection, loco, study_alloc = combine_allocation(recs)
    rep = audit_stage1(records=recs, selection=selection, loco=loco, study_alloc=study_alloc,
                       source_checksums={}, expected_policy_hash="deadbeef")
    assert not rep.policy_hash_reproduced and rep.decision.startswith("STOP")


def test_audit_rechecks_floors_after_global_dedup():
    # igsf sequences collapse to 3 distinct aa -> global_dedup leaves only 3 igsf records -> floor STOP.
    recs = _cohort()
    for i, r in enumerate([x for x in recs if x["group"] == "igsf"]):
        r["aa"] = f"IGSFDUP{i % 3}"
    selection, loco, study_alloc = combine_allocation(recs)
    kept, dropped = global_dedup(recs)
    rep = audit_stage1(records=kept, selection=selection, loco=loco, study_alloc=study_alloc,
                       source_checksums={},
                       required_groups={"igk_igl", "vhh_nonhuman", "igsf", "positive_real"})
    assert rep.decision.startswith("STOP")
    assert any(f.startswith("floor igsf") for f in rep.failures), rep.failures


def test_locus_census_not_polluted_by_structural_shared_lineage():
    # structural records share their parent's lineage_key; the per-locus census must still report the
    # parent's true category (record-based count), not be overwritten by a structural category.
    from alignair.aa.guard.structural import structural_records
    parent = {"group": "positive_real", "category": "pos_igh_repertoire",
              "aa": "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG",
              "lineage_key": "S::d::IGHV3-23", "repository": "oas", "study": "S_2020", "donor": "d"}
    struct = structural_records(parent, role_of_parent="selection")
    selection = {"positive_real": {parent["lineage_key"]}, "structural": {parent["lineage_key"]}}
    rep = audit_stage1(records=[parent] + struct, selection=selection, loco={}, study_alloc=None,
                       source_checksums={}, required_groups=set())
    assert rep.counts_by_locus["selection"].get("pos_igh_repertoire", 0) == 1
    assert rep.counts_by_locus["selection"].get("?", 0) == 0
    assert rep.counts_by_role["selection"].get("structural") == 1   # 6 transforms share 1 parent lineage


def test_same_donor_positive_real_straddle_detected_despite_structural():
    # Integrity: same study+donor, two DIFFERENT lineages split across roles must flag study+donor overlap
    # even though each parent has structural children that share its lineage_key and carry no study field.
    from alignair.aa.guard.structural import structural_records
    p1 = {"group": "positive_real", "category": "pos_igh_repertoire", "aa": "QVQLVQSGAEVKKPGSSVKVSCKAS",
          "lineage_key": "X_2020::d::IGHV1", "repository": "oas", "study": "X_2020", "donor": "d"}
    p2 = {**p1, "aa": "EVQLVESGGGLVQPGGSLRLSCAAS", "lineage_key": "X_2020::d::IGHV3"}
    s1 = structural_records(p1, role_of_parent="selection")
    s2 = structural_records(p2, role_of_parent="loco")
    selection = {"positive_real": {p1["lineage_key"]}, "structural": {p1["lineage_key"]}}
    loco = {"positive_real": {p2["lineage_key"]}, "structural": {p2["lineage_key"]}}
    rep = audit_stage1(records=[p1, p2] + s1 + s2, selection=selection, loco=loco, study_alloc=None,
                       source_checksums={}, required_groups=set())
    assert rep.role_overlap["study"] >= 1 and rep.role_overlap["donor"] >= 1
    assert rep.decision.startswith("STOP")
    assert rep.counts_by_study == {"selection": 1, "loco": 1}   # study visible, not masked by structural


def test_audit_flags_single_donor_igl_slice():
    # A validation (loco) IGL slice resting on ONE donor/study must trip the diversity gate even though
    # its lineage count clears the 37 floor. Selection here is diverse and must pass.
    sel_recs = ([_oas("SelA_2020", f"a{i}", "igl", i) for i in range(40)] +
                [_oas("SelB_2020", f"b{i}", "igl", i) for i in range(40)] +
                [_oas("SelC_2020", f"c{i}", "igl", i) for i in range(40)])
    loco_recs = [_oas("LocoX_2021", "solo", "igl", i) for i in range(40)]
    records = sel_recs + loco_recs
    selection = {"igk_igl": {r["lineage_key"] for r in sel_recs}}
    loco = {"igk_igl": {r["lineage_key"] for r in loco_recs}}
    rep = audit_stage1(records=records, selection=selection, loco=loco, study_alloc=None,
                       source_checksums={}, required_groups=set())
    assert rep.locus_diversity["igl"]["loco"] == {"donors": 1, "studies": 1, "ok": False}
    assert rep.locus_diversity["igl"]["selection"]["ok"] is True
    assert any(f.startswith("locus_diversity igl loco") for f in rep.failures)
    assert rep.decision.startswith("STOP")


def test_global_dedup_removes_cross_source_duplicates():
    recs = [_igsf(1), {**_igsf(1), "lineage_key": "igsf-other"}]  # same aa, different lineage
    kept, dropped = global_dedup(recs)
    assert dropped == 1 and len(kept) == 1
