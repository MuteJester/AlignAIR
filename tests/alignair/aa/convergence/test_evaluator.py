import numpy as np
import pytest

from alignair.aa.convergence.evaluator import STRATA_ORDER, build_cohort, evaluate_checkpoint
from alignair.aa.strategic import strategic_setup

DC_NAME = "HUMAN_IGH_OGRDB"
SEED = 20250722          # disjoint from calibrate_stage2's 1000/9000 and training seed 0
N_PER_STRATUM = 40       # small but big enough that D/truncate/short_janchor buckets are non-empty

REQUIRED_STRATA = ("clean", "aa_truncate_50_80", "short_janchor", "moderate_full")


@pytest.fixture(scope="module")
def cohort():
    coh, fp = build_cohort(DC_NAME, n_per_stratum=N_PER_STRATUM, seed=SEED)
    return coh, fp


def test_build_cohort_fingerprint_is_deterministic():
    coh_a, fp_a = build_cohort(DC_NAME, n_per_stratum=N_PER_STRATUM, seed=SEED)
    coh_b, fp_b = build_cohort(DC_NAME, n_per_stratum=N_PER_STRATUM, seed=SEED)
    assert fp_a == fp_b
    assert isinstance(fp_a, str) and len(fp_a) == 64          # sha256 hex digest

    for name in REQUIRED_STRATA:
        assert coh_a["by_stratum"].get(name), f"missing/empty required stratum: {name}"
    assert coh_a["calibration"], "calibration slice must be non-empty"

    # calibration is disjoint from the "clean" eval rows (no shared record identity)
    cal_ids = {id(r["record"]) for r in coh_a["calibration"]}
    clean_ids = {id(r["record"]) for r in coh_a["by_stratum"]["clean"]}
    assert cal_ids.isdisjoint(clean_ids)


def test_build_cohort_fingerprint_changes_with_seed():
    _, fp_a = build_cohort(DC_NAME, n_per_stratum=N_PER_STRATUM, seed=SEED)
    _, fp_b = build_cohort(DC_NAME, n_per_stratum=N_PER_STRATUM, seed=SEED + 1)
    assert fp_a != fp_b


def test_build_cohort_strata_are_subset_of_declared_order(cohort):
    coh, _fp = cohort
    assert set(coh["by_stratum"].keys()) <= set(STRATA_ORDER)


def test_moderate_full_is_its_own_stratum_not_merged_into_clean(cohort):
    """`clean` must be clean_full-only (matches the frozen baseline/§5 gates); moderate_full
    (n_indels==0 rows from the moderate_full GenAIRR stratum) is scored SEPARATELY and must never
    be pooled into `clean`."""
    coh, _fp = cohort
    assert "moderate_full" in coh["by_stratum"]
    assert coh["by_stratum"]["moderate_full"], "moderate_full stratum must be non-empty"

    clean_ids = {id(r["record"]) for r in coh["by_stratum"]["clean"]}
    moderate_ids = {id(r["record"]) for r in coh["by_stratum"]["moderate_full"]}
    assert clean_ids.isdisjoint(moderate_ids)


def test_evaluate_checkpoint_well_formed(cohort):
    coh, _fp = cohort
    model, cfg, class_maps, meta = strategic_setup(DC_NAME, 192)
    model.eval()

    out = evaluate_checkpoint(model, cfg, class_maps, meta, coh, device="cpu")

    required_top_level = (
        "clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
        "d_allele_acc", "d_allele_macro", "d_gene_acc",
        "by_stratum", "freq_bins", "boundary_mae", "nll", "brier", "conformal",
    )
    for key in required_top_level:
        assert key in out, f"evaluate_checkpoint output missing key: {key}"

    accuracy_keys = ("clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
                     "d_allele_acc", "d_allele_macro", "d_gene_acc")
    for key in accuracy_keys:
        val = out[key]
        assert val is not None, f"{key} unexpectedly None (untrained model, tiny cohort)"
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    # by_stratum: required strata present with well-formed v_acc
    for name in REQUIRED_STRATA:
        assert name in out["by_stratum"], f"by_stratum missing required stratum: {name}"
    for name, entry in out["by_stratum"].items():
        assert entry["n"] > 0
        assert 0.0 <= entry["v_acc"] <= 1.0

    # conformal: coverage in [0,1], set sizes well-formed (>=0, p95 >= mean)
    conf = out["conformal"]
    assert conf["coverage"] is not None
    assert 0.0 <= conf["coverage"] <= 1.0
    assert conf["set_mean"] >= 0.0
    assert conf["set_p95"] >= 0.0

    # nll / brier for v, j, d: present and non-negative
    for gene in ("v", "j", "d"):
        assert out["nll"][gene] is not None and out["nll"][gene] >= 0.0
        assert out["brier"][gene] is not None and out["brier"][gene] >= 0.0

    # informational metrics: keys must exist (best-effort; values may legitimately be partial)
    assert set(out["freq_bins"].keys()) == {"v", "j"}
    assert "v" in out["boundary_mae"] and "j" in out["boundary_mae"]


def test_evaluate_checkpoint_is_a_pure_function_of_model_and_cohort(cohort):
    """Same frozen model + same frozen cohort -> identical metric dict (no hidden randomness)."""
    coh, _fp = cohort
    model, cfg, class_maps, meta = strategic_setup(DC_NAME, 192)
    model.eval()

    out_a = evaluate_checkpoint(model, cfg, class_maps, meta, coh, device="cpu")
    out_b = evaluate_checkpoint(model, cfg, class_maps, meta, coh, device="cpu")

    assert out_a["clean_v_acc"] == out_b["clean_v_acc"]
    assert out_a["clean_j_acc"] == out_b["clean_j_acc"]
    assert out_a["conformal"]["coverage"] == out_b["conformal"]["coverage"]
