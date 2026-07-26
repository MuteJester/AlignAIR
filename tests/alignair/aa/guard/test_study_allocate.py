from alignair.aa.guard.study_allocate import assign_studies, canonical_study


def _rec(study, subject, category, group, lk, repo="oas"):
    return {"repository": repo, "study": study, "donor": subject, "category": category,
            "group": group, "lineage_key": lk}


def test_canonical_study_collapses_aliases():
    a = canonical_study("oas", "Bernat_2019_1")
    b = canonical_study("oas", "Bernat_2019_2")
    c = canonical_study("oas", "Bernat et al., 2019")
    assert a == b == c
    # repository namespaces studies apart
    assert canonical_study("vdjserver", "Bernat_2019_1") != a


def test_whole_study_never_split_across_roles():
    recs = []
    for st in range(10):
        for i in range(40):
            recs.append(_rec(f"Study{st}_2020", f"subj{st}_{i}", "igk", "igk_igl", f"{st}-K{i}"))
            recs.append(_rec(f"Study{st}_2020", f"subj{st}_{i}", "igl", "igk_igl", f"{st}-L{i}"))
    a = assign_studies(recs)
    # every record of a study shares that study's role -> no study straddles selection & loco
    for group in ("igk_igl",):
        assert not (a.selection.get(group, set()) & a.loco_challenge.get(group, set()))
    # role_of_study assigns exactly one role per canonical study
    assert set(a.role_of_study.values()) <= {"selection", "loco_challenge"}


def test_heavy_and_light_from_same_study_inherit_same_role():
    recs = []
    for st in range(8):
        for i in range(40):
            recs.append(_rec(f"S{st}_2020", f"d{st}_{i}", "igk", "igk_igl", f"{st}-K{i}"))
            recs.append(_rec(f"S{st}_2020", f"d{st}_{i}", "igl", "igk_igl", f"{st}-L{i}"))
            recs.append(_rec(f"S{st}_2020", f"d{st}_{i}", "pos_igh_repertoire", "positive_real", f"{st}-H{i}"))
    a = assign_studies(recs)
    cs = canonical_study("oas", "S0_2020")
    role = a.role_of_study[cs]
    other = "loco_challenge" if role == "selection" else "selection"
    # S0's heavy lineages are in its role and NOT the other role
    assert f"0-H0" in a.selection.get("positive_real", set()) | a.loco_challenge.get("positive_real", set())
    assert f"0-H0" not in (a.loco_challenge if role == "selection" else a.selection).get("positive_real", set())


def test_balances_toward_locus_floors():
    # 12 studies each with 40 igk + 40 igl lineages -> both roles should clear 37/37
    recs = []
    for st in range(12):
        for i in range(40):
            recs.append(_rec(f"St{st}_2019", f"p{st}_{i}", "igk", "igk_igl", f"{st}-K{i}"))
            recs.append(_rec(f"St{st}_2019", f"p{st}_{i}", "igl", "igk_igl", f"{st}-L{i}"))
    a = assign_studies(recs)
    assert a.locus_min_ok["igk_igl"]["selection"] and a.locus_min_ok["igk_igl"]["loco_challenge"]


def test_policy_hash_is_frozen_and_deterministic():
    recs = [_rec("A_2020", "d1", "igk", "igk_igl", "K1"), _rec("B_2021", "d2", "igl", "igk_igl", "L1")]
    a1 = assign_studies(recs)
    a2 = assign_studies(recs)
    assert a1.policy_hash == a2.policy_hash and len(a1.policy_hash) == 64
