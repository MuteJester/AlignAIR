from alignair.aa.guard.allocate import allocate_fresh, GUARDBAND_FLOOR


def test_selection_and_loco_never_share_a_lineage():
    recs = [{"group": "igk_igl", "category": "igk", "lineage_key": f"L{i}"} for i in range(200)]
    a = allocate_fresh(recs)
    assert not (a.selection["igk_igl"] & a.loco_challenge["igk_igl"])
    assert a.pooled_overlap == set()


def test_guardband_floor_values():
    assert GUARDBAND_FLOOR["gate_5pct"] == 74 and GUARDBAND_FLOOR["gate_10pct"] == 36


def test_vhh_min_19_enforced_in_each_subset():
    recs = ([{"group": "vhh_nonhuman", "category": "vhh", "lineage_key": f"V{i}",
              "heavy_species": "lama glama", "deposition_date": "2020-01-01"} for i in range(50)]
            + [{"group": "vhh_nonhuman", "category": "murine_vh", "lineage_key": f"M{i}"} for i in range(40)])
    a = allocate_fresh(recs)
    assert a.vhh_min_ok["selection"] and a.vhh_min_ok["loco_challenge"]


def test_all_lama_input_falls_back_to_sha_rank_split():
    # No genus diversity -> strict genus holdout cannot fill the challenge side -> fallback, recorded.
    recs = [{"group": "vhh_nonhuman", "category": "vhh", "lineage_key": f"V{i}",
             "heavy_species": "lama glama", "deposition_date": "2020-01-01"} for i in range(50)]
    a = allocate_fresh(recs)
    assert a.vhh_mode == "fallback_sha_rank"


def test_genus_and_temporal_diversity_uses_strict_holdout():
    # L. glama pre-cutoff -> selection; vicugna/camelus post-cutoff -> challenge; each >= 19.
    sel_side = [{"group": "vhh_nonhuman", "category": "vhh", "lineage_key": f"S{i}",
                 "heavy_species": "lama glama", "deposition_date": "2018-01-01"} for i in range(25)]
    chal_side = [{"group": "vhh_nonhuman", "category": "vhh", "lineage_key": f"C{i}",
                  "heavy_species": "vicugna pacos", "deposition_date": "2024-01-01"} for i in range(25)]
    a = allocate_fresh(sel_side + chal_side, vhh_temporal_cutoff="2020-01-01")
    assert a.vhh_mode == "genus_temporal_holdout"
    # selection VHH are the L. glama pre-cutoff; challenge VHH are the vicugna post-cutoff
    assert {f"S{i}" for i in range(25)} <= a.selection["vhh_nonhuman"]
    assert {f"C{i}" for i in range(25)} <= a.loco_challenge["vhh_nonhuman"]
    assert not (a.selection["vhh_nonhuman"] & a.loco_challenge["vhh_nonhuman"])
