import json

from scripts.aa.applicability.hydrate_fresh import run_stage1, stage1_decision


def _all_ok():
    return dict(
        floors_ok={"igk_igl": True, "tcr": True, "vhh_nonhuman": True, "igsf": True,
                   "structural": True, "positive_real": True, "positive_sim": True},
        vhh_min_ok={"selection": True, "loco_challenge": True},
    )


def test_go_requires_floors_vhh_and_disjoint():
    ok = _all_ok()
    assert stage1_decision(ok, clean=True) == "GO"
    ok["floors_ok"]["tcr"] = False
    assert stage1_decision(ok, clean=True).startswith("STOP")
    ok["floors_ok"]["tcr"] = True
    assert stage1_decision(ok, clean=False).startswith("STOP")


def test_stop_when_vhh_min_unmet():
    ok = _all_ok()
    ok["vhh_min_ok"]["loco_challenge"] = False
    assert stage1_decision(ok, clean=True).startswith("STOP")


def test_run_stage1_stops_on_disjointness_overlap(tmp_path):
    # A fresh record colliding with the Gate-3 denylist -> audit dirty -> STOP, no allocation trusted.
    import hashlib
    seq = "QVQLVQSGAEVKK"
    h = hashlib.sha256(seq.encode()).hexdigest()
    fresh = [{"id": "x", "aa": seq, "group": "igk_igl", "category": "igk", "lineage_key": "L1"}]
    decision = run_stage1(
        fresh,
        gate3_denylist={"sequence_hashes": {h}, "lineages": set(), "donors": set(), "studies": set()},
        phase3=({}, {}),
        vhh_temporal_cutoff=None,
        private_dir=tmp_path / "priv",
        artifact_dir=tmp_path / "art",
    )
    assert decision.startswith("STOP")
    result = json.loads((tmp_path / "art" / "fresh_cohort_census.json").read_text())
    assert result["decision"].startswith("STOP")
    assert result["disjoint_clean"] is False


def test_run_stage1_go_on_well_powered_disjoint_cohort(tmp_path):
    # Enough disjoint lineages per group to clear guardband floors (74 for 5% gates, 36 for 10% gates).
    fresh = []
    for g, n in (("igk_igl", 200), ("tcr", 200), ("igsf", 120), ("structural", 120),
                 ("positive_real", 120), ("positive_sim", 120)):
        cat = {"igk_igl": "igk", "tcr": "tra"}.get(g, g)
        fresh += [{"id": f"{g}{i}", "aa": f"SEQ{g}{i}", "group": g, "category": cat,
                   "lineage_key": f"{g}-L{i}"} for i in range(n)]
    # vhh_nonhuman: 60 genus-diverse VHH (30 lama pre-cutoff, 30 vicugna post-cutoff) + 80 murine
    fresh += [{"id": f"vs{i}", "aa": f"VHHs{i}", "group": "vhh_nonhuman", "category": "vhh",
               "lineage_key": f"vhh-S{i}", "heavy_species": "lama glama",
               "deposition_date": "2018-01-01"} for i in range(30)]
    fresh += [{"id": f"vc{i}", "aa": f"VHHc{i}", "group": "vhh_nonhuman", "category": "vhh",
               "lineage_key": f"vhh-C{i}", "heavy_species": "vicugna pacos",
               "deposition_date": "2024-01-01"} for i in range(30)]
    fresh += [{"id": f"m{i}", "aa": f"MUR{i}", "group": "vhh_nonhuman", "category": "murine_vh",
               "lineage_key": f"mur-L{i}"} for i in range(80)]
    decision = run_stage1(
        fresh,
        gate3_denylist={"sequence_hashes": set(), "lineages": set(), "donors": set(), "studies": set()},
        phase3=({}, {}),
        vhh_temporal_cutoff="2020-01-01",
        private_dir=tmp_path / "priv",
        artifact_dir=tmp_path / "art",
    )
    assert decision == "GO"
    result = json.loads((tmp_path / "art" / "fresh_cohort_census.json").read_text())
    assert result["vhh_mode"] == "genus_temporal_holdout"
    # private allocation written; artifact carries counts only (no sequences)
    assert (tmp_path / "priv" / "allocation.json").exists()
    assert "aa" not in json.dumps(result)
