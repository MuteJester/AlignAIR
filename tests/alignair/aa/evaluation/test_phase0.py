import pytest

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_build_phase0_report_smoke_small_n():
    from alignair.aa.phase0 import build_phase0_report
    r = build_phase0_report(n=200, seed=0, rate=0.05, productive_only=True)
    assert r["schema"] == "alignair.aa.phase0.v1"
    # identifiability computed for V and J from real germlines
    assert r["identifiability"]["V"]["n_alleles"] == 198
    assert r["identifiability"]["J"]["n_alleles"] == 7
    assert 0.0 <= r["identifiability"]["V"]["frac_alleles_uniquely_identifiable"] <= 1.0
    # simulation-derived stats present and sane
    assert r["length_stats"]["n_records"] > 0
    assert 0.0 <= r["boundary_stats"]["V_end_cut_frac"] <= 1.0
    assert r["d_stats"]["mean_owned_d_residues"] >= 0.0
    assert r["meta"]["db_hash"] and len(r["meta"]["db_hash"]) == 16
