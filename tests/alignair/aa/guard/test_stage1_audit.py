import hashlib

from alignair.aa.guard.stage1_audit import audit_stage1, combine_allocation, global_dedup


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


def test_global_dedup_removes_cross_source_duplicates():
    recs = [_igsf(1), {**_igsf(1), "lineage_key": "igsf-other"}]  # same aa, different lineage
    kept, dropped = global_dedup(recs)
    assert dropped == 1 and len(kept) == 1
