import pytest
from alignair.aa.guard.fresh_manifest import FRESH_MANIFEST, FreshCategorySpec

pytestmark = pytest.mark.component


def test_manifest_covers_all_binding_groups():
    groups = {c.group for c in FRESH_MANIFEST}
    assert {"igk_igl", "tcr", "vhh_nonhuman", "igsf", "structural",
            "positive_real", "positive_sim"} <= groups


def test_vhh_declares_genus_and_temporal_holdout():
    vhh = [c for c in FRESH_MANIFEST if c.key == "vhh"]
    assert vhh, "vhh category present"
    v = vhh[0]
    assert v.vhh_genus_holdout is not None and "selection" in v.vhh_genus_holdout
    assert "pre-split Gate-3 VHH" in v.vhh_temporal_cutoff_policy
    assert "SAbDab" in v.source and "single-source" in v.note.lower()


def test_all_specs_are_frozen_dataclasses():
    for c in FRESH_MANIFEST:
        assert isinstance(c, FreshCategorySpec)
        assert c.order == "N->C"
