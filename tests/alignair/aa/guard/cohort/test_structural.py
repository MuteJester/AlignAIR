import pytest
from alignair.aa.guard.structural import make_structural, structural_records

pytestmark = pytest.mark.component


PARENT_AA = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDLYYYGMDVWGQGTTVTVSS"
CDR3 = "ARDLYYYGMDV"


def test_all_six_transforms_present():
    t = make_structural(PARENT_AA, CDR3)
    assert set(t) == {"reversed_vh", "v_only", "j_only", "cdr3_only", "leader_vh", "vh_constant"}


def test_reversed_is_reverse_and_others_derive_from_parent():
    t = make_structural(PARENT_AA, CDR3)
    assert t["reversed_vh"] == PARENT_AA[::-1]
    assert t["cdr3_only"] == CDR3
    assert t["leader_vh"].endswith(PARENT_AA) and len(t["leader_vh"]) > len(PARENT_AA)
    assert t["vh_constant"].startswith(PARENT_AA) and len(t["vh_constant"]) > len(PARENT_AA)


def test_records_inherit_parent_lineage_and_role():
    parent = {"aa": PARENT_AA, "cdr3_aa": CDR3, "lineage_key": "Study::Subj::IGHV1::ARDLY",
              "group": "positive_real", "category": "pos_igh_repertoire"}
    recs = structural_records(parent, role_of_parent="loco")
    # every structural record carries the PARENT lineage (parent-level scoring) + inherited role
    assert all(r["lineage_key"] == parent["lineage_key"] for r in recs)
    assert all(r["parent_lineage"] == parent["lineage_key"] for r in recs)
    assert all(r["inherited_role"] == "loco" for r in recs)
    assert all(r["group"] == "structural" for r in recs)
    # reasons match the completeness taxonomy
    reasons = {r["category"]: r["expected_reason"] for r in recs}
    assert reasons["reversed_vh"] == "reversed_order"
    assert reasons["cdr3_only"] == "too_short"


def test_only_real_parents_allowed():
    import pytest
    sim = {"aa": PARENT_AA, "cdr3_aa": CDR3, "lineage_key": "genairr_fresh_seed_1", "group": "positive_sim"}
    with pytest.raises(ValueError):
        structural_records(sim, role_of_parent="selection")
