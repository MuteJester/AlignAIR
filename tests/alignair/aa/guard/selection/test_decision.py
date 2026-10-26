import pytest
from types import SimpleNamespace

from alignair.aa.guard.decision import (
    CANDIDATE_DECISION_COMPONENTS, CandidateSpec, apply_decision, candidate_grid, grid_keys,
)

pytestmark = pytest.mark.component


def _rs(*, accept=True, within=True, energy=-10.0):
    comp = (5.0, 0, 0, 0, 0) if accept else (0, 5.0, 0, 0, 0)          # P(complete) high vs low
    typ = (5.0, 0, 0, 0, 0, 0, 0, 0) if accept else (0, 5.0, 0, 0, 0, 0, 0, 0)
    hmm = ({"has_v": True, "has_j": True, "order_ok": True, "combined_coverage": 0.95,
            "best_v_group": "human_igh", "best_alt_score": 0.0, "human_igh_score": 30.0} if accept else
           {"has_v": True, "has_j": True, "order_ok": True, "combined_coverage": 0.95,
            "best_v_group": "human_igl", "best_alt_score": 30.0, "human_igh_score": 0.0})
    return SimpleNamespace(within_length_contract=within, hmm_evidence=hmm, completeness_logits=comp,
                           domain_type_logits=typ, applicability_logit=(5.0 if accept else -5.0), energy=energy)


def test_candidate_grids_are_finite_and_candidate_specific():
    assert len(candidate_grid("C0")) == 2                             # hmm only (1 coverage x 2 margin)
    assert len(candidate_grid("C1")) == 24 and len(candidate_grid("C2")) == 24
    assert len(candidate_grid("C3")) == 48 and len(candidate_grid("C4")) == 48   # + energy dimension
    assert "energy_max" in grid_keys("C3") and "energy_max" not in grid_keys("C1")
    assert "applicability_logit_cutoff" not in grid_keys("C0")        # C0 is HMM-only


def test_every_active_component_appears_in_the_grid():
    for cid, comps in CANDIDATE_DECISION_COMPONENTS.items():
        keys = set(grid_keys(cid))
        if "applicability_logit" in comps:
            assert "applicability_logit_cutoff" in keys
        if "energy" in comps:
            assert "energy_max" in keys
        if "completeness" in comps:
            assert "completeness_min_prob" in keys
        if "human_igh_type" in comps:
            assert "human_igh_type_min_prob" in keys


def test_conjunctive_accept_and_single_component_failure():
    gp = candidate_grid("C3")[0]                                      # most permissive point
    assert apply_decision(_rs(accept=True), CandidateSpec.of("C3"), gp).accepted
    # flip ONLY the applicability logit below cutoff -> conjunction fails on that one component
    r = _rs(accept=True); r.applicability_logit = -1.0
    d = apply_decision(r, CandidateSpec.of("C3"), dict(gp, applicability_logit_cutoff=1.0))
    assert not d.accepted and "applicability_logit" in d.failing_components


def test_energy_only_binds_for_c3_c4():
    r = _rs(accept=True); r.energy = 100.0                            # very high energy = out of domain
    strict = dict(candidate_grid("C3")[0], energy_max=-5.0)
    assert not apply_decision(r, CandidateSpec.of("C3"), strict).accepted     # energy gate rejects
    # C1 ignores energy entirely
    assert apply_decision(r, CandidateSpec.of("C1"), candidate_grid("C1")[0]).accepted


def test_c0_uses_only_hmm_evidence_no_learned_logits_needed():
    r = SimpleNamespace(within_length_contract=True, applicability_logit=None, energy=None,
                        completeness_logits=None, domain_type_logits=None,
                        hmm_evidence={"has_v": True, "has_j": True, "order_ok": True, "combined_coverage": 0.95,
                                      "best_v_group": "human_igh", "best_alt_score": 0.0, "human_igh_score": 30.0})
    assert apply_decision(r, CandidateSpec.of("C0"), candidate_grid("C0")[0]).accepted


def test_over_length_is_never_supported():
    d = apply_decision(_rs(accept=True, within=False), CandidateSpec.of("C2"), candidate_grid("C2")[0])
    assert not d.accepted and d.failing_components == ("length_contract",)
