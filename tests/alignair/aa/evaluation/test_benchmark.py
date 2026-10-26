import pytest

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_run_aa_benchmark_stratifies_and_flags_non_aa_strata():
    from alignair_benchmark import default_igh_spec

    from alignair.aa.benchmark import run_aa_benchmark
    from alignair.aa.data import build_class_maps
    from alignair.aa.train_gym import setup_aa

    model, ref, cfg, _ = setup_aa("HUMAN_IGH_OGRDB", 192)      # untrained is fine: no_aa_rate is data-driven
    cmaps, _ = build_class_maps("HUMAN_IGH_OGRDB")
    eval_class_maps = {"V": cmaps["V"].allele_class, "J": cmaps["J"].allele_class}

    spec = default_igh_spec(n_per_stratum=10, seed=1)
    r = run_aa_benchmark(model, ref, cfg, eval_class_maps, spec=spec, device="cpu", batch_size=64)

    bs = r["by_stratum"]
    # full-length strata keep the junction -> AA available
    assert bs["clean_full"]["no_aa_rate"] < 0.2
    # V-keeping FR amplicons trim the 3'/junction -> no sequence_aa -> NOT AA-evaluable
    assert bs["amplicon_fr1"]["no_aa_rate"] > 0.8
    assert bs["amplicon_fr2"]["no_aa_rate"] > 0.8
    # J-anchored amplicons KEEP the junction (they trim the 5'/V) -> AA available (but V is gone)
    assert bs["amplicon_janchor"]["no_aa_rate"] < 0.3
    # RC / non-forward strata are excluded (proteins have no reverse complement)
    assert bs["amplicon_janchor_rc"]["n_non_forward"] > 0
    # per-stratum allele metric structure
    for g in ("v", "d", "j"):
        assert "nt_top1" in bs["clean_full"]["alleles"][g]
    assert "overall_alleles" in r and "by_stratum" in r
