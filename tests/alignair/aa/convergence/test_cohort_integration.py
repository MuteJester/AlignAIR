"""Live GenAIRR integration contracts for convergence cohort construction."""
import pytest
from alignair.aa.convergence.evaluator import STRATA_ORDER, build_cohort

pytestmark = pytest.mark.integration

DC_NAME = "HUMAN_IGH_OGRDB"
SEED = 20250722
N_PER_STRATUM = 40
REQUIRED_STRATA = ("clean", "aa_truncate_50_80", "short_janchor", "moderate_full")

@pytest.fixture
def cohort():
    return build_cohort(DC_NAME, n_per_stratum=N_PER_STRATUM, seed=SEED)

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
