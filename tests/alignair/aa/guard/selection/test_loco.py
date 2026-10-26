import pytest

from alignair.aa.guard.loco import (LOCO_FOLDS, LOCO_MASK_HASH, LOCO_MASK_MAP, assert_no_g_leak,
                                    assert_reason_blind, loco_exclude, masked_profiles_for)

pytestmark = pytest.mark.component


def test_mask_map_is_biologically_complete_and_hashed():
    assert LOCO_MASK_MAP["igk_igl"] == ("human_igk", "human_igl")
    assert LOCO_MASK_MAP["tcr"] == ("tra", "trb", "trg", "trd")
    assert LOCO_MASK_MAP["vhh_nonhuman"] == ("camelid_igh", "murine_vh")   # NOT the stale 'vhh'
    assert LOCO_MASK_MAP["igsf"] == ("igsf_vset",)
    assert set(LOCO_FOLDS) == {"igk_igl", "tcr", "vhh_nonhuman", "igsf"}
    assert len(LOCO_MASK_HASH) == 64


def test_loco_exclude_removes_only_fold_group():
    cohort = [{"id": "a", "group": "igk_igl"}, {"id": "b", "group": "positive_real"},
              {"id": "c", "group": "igk_igl"}, {"id": "d", "group": "tcr"}]
    out = loco_exclude(cohort, "igk_igl")
    assert {r["id"] for r in out} == {"b", "d"}


def test_assert_no_g_leak():
    assert_no_g_leak(["t1", "t2"], ["h1"], ["x"])           # no leak -> ok
    with pytest.raises(ValueError):
        assert_no_g_leak(["t1", "g1"], ["h1"], ["g1"])      # g1 in train -> raise
    with pytest.raises(ValueError):
        assert_no_g_leak(["t1"], ["g2"], ["g2"])            # g2 in threshold -> raise


def test_assert_reason_blind():
    assert_reason_blind("human_igh_supported", masked_profiles_for("igk_igl"))   # ok
    with pytest.raises(ValueError):
        assert_reason_blind("alt_locus:human_igk", masked_profiles_for("igk_igl"))
    with pytest.raises(ValueError):
        assert_reason_blind("alt_locus:camelid_igh", masked_profiles_for("vhh_nonhuman"))
