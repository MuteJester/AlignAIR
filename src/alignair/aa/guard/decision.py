"""Stage-5 Task 5.3 — the FROZEN conjunctive candidate decision graph + a PURE ``apply_decision``.

The approved guard decision is conjunctive over decision components; a record is accepted as ``supported``
only if EVERY active component passes. Which components are active is candidate-specific:

  * HMM evidence            — complete-domain geometry + human-IGH wins its locus by margin (ALL candidates, incl. C0)
  * Completeness evidence   — the learned 5-way head says ``complete`` (C1–C4)
  * Human-IGH type evidence — the learned 8-way head says ``human_igh_vh`` (C1–C4)
  * Applicability logit      — the learned binary head clears a cutoff (C1–C4)
  * Energy, when present     — open-set free energy is low enough (C3–C4 only)

``apply_decision`` is a pure function of a frozen ``RecordScore`` (raw outputs), a ``CandidateSpec``, and a
finite grid point. It reruns NO inference — a frozen threshold can be re-evaluated entirely from the artifact.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from alignair.aa.guard.classes import COMPLETE, COMPLETENESS_INDEX, DOMAIN_TYPE_INDEX, SUPPORTED_DOMAIN_TYPE

# Which decision components each candidate's conjunctive graph uses (C0 is the HMM-only rule baseline).
CANDIDATE_DECISION_COMPONENTS = {
    "C0": ("hmm",),
    "C1": ("hmm", "completeness", "human_igh_type", "applicability_logit"),
    "C2": ("hmm", "completeness", "human_igh_type", "applicability_logit"),
    "C3": ("hmm", "completeness", "human_igh_type", "applicability_logit", "energy"),
    "C4": ("hmm", "completeness", "human_igh_type", "applicability_logit", "energy"),
}

# Accept condition per grid cutoff key (frozen score directions).
COMPONENT_DIRECTIONS = {
    "hmm_min_coverage": "geometry_prerequisite (combined_coverage >= cut)",
    "hmm_human_margin": "higher_is_more_human_igh (human_igh_score - best_alt_score >= cut)",
    "completeness_min_prob": "higher_is_more_complete (P(complete) >= cut)",
    "human_igh_type_min_prob": "higher_is_more_human_igh_vh (P(human_igh_vh) >= cut)",
    "applicability_logit_cutoff": "higher_is_more_supported (logit >= cut)",
    "energy_max": "lower_is_more_in_domain (energy <= cut)",
}

# Finite ordered candidate levels per cutoff key (permissive -> restrictive).
COMPONENT_LEVELS = {
    "hmm_min_coverage": (0.80,),
    "hmm_human_margin": (0.0, 5.0),
    "completeness_min_prob": (0.5, 0.9),
    "human_igh_type_min_prob": (0.5, 0.9),
    "applicability_logit_cutoff": (0.0, 1.0, 2.0),
    "energy_max": (0.0, -5.0),                 # energy = -logsumexp(type); more negative = stricter in-domain
}

# Each decision component maps to the finite grid cutoff keys it sweeps.
COMPONENT_GRID_KEYS = {
    "hmm": ("hmm_min_coverage", "hmm_human_margin"),
    "completeness": ("completeness_min_prob",),
    "human_igh_type": ("human_igh_type_min_prob",),
    "applicability_logit": ("applicability_logit_cutoff",),
    "energy": ("energy_max",),
}


def grid_keys(cid: str) -> tuple:
    """Ordered cutoff keys active for a candidate (component order, then key order within a component)."""
    return tuple(k for comp in CANDIDATE_DECISION_COMPONENTS[cid] for k in COMPONENT_GRID_KEYS[comp])


def candidate_grid(cid: str) -> tuple:
    """The finite, ordered, candidate-specific conjunctive grid (Cartesian product of the active cutoff
    levels, permissive-first). Sizes: C0=2, C1/C2=24, C3/C4=48 points."""
    keys = grid_keys(cid)
    levels = [COMPONENT_LEVELS[k] for k in keys]
    return tuple(dict(zip(keys, combo)) for combo in itertools.product(*levels))


@dataclass(frozen=True)
class CandidateSpec:
    cid: str
    components: tuple

    @classmethod
    def of(cls, cid: str) -> "CandidateSpec":
        return cls(cid, CANDIDATE_DECISION_COMPONENTS[cid])


@dataclass(frozen=True)
class Decision:
    accepted: bool                 # accepted == "supported"
    state: str                     # "supported" | "not_supported"
    failing_components: tuple      # components that failed (empty iff accepted)
    per_component: dict            # component -> bool pass


def _softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _component_pass(comp: str, rs, tp: dict) -> bool:
    h = rs.hmm_evidence
    if comp == "hmm":
        geometry = bool(h["has_v"] and h["has_j"] and h["order_ok"]
                        and h["combined_coverage"] >= tp["hmm_min_coverage"])
        human_wins = h["best_v_group"] == "human_igh"
        margin_ok = (h["human_igh_score"] - h["best_alt_score"]) >= tp["hmm_human_margin"]
        return bool(geometry and human_wins and margin_ok)
    if comp == "completeness":
        return _softmax(rs.completeness_logits)[COMPLETENESS_INDEX[COMPLETE]] >= tp["completeness_min_prob"]
    if comp == "human_igh_type":
        return _softmax(rs.domain_type_logits)[DOMAIN_TYPE_INDEX[SUPPORTED_DOMAIN_TYPE]] >= tp["human_igh_type_min_prob"]
    if comp == "applicability_logit":
        return rs.applicability_logit >= tp["applicability_logit_cutoff"]
    if comp == "energy":
        return rs.energy <= tp["energy_max"]
    raise KeyError(f"unknown decision component: {comp!r}")


def apply_decision(record_score, candidate_spec: CandidateSpec, threshold_point: dict) -> Decision:
    """PURE conjunctive decision from frozen raw outputs — no inference rerun.

    An input outside the length contract is never supported (the no-truncation withholding contract). Otherwise
    the record is ``supported`` iff every active component clears its cutoff in ``threshold_point``."""
    if not record_score.within_length_contract:
        return Decision(False, "not_supported", ("length_contract",), {"length_contract": False})
    per = {comp: _component_pass(comp, record_score, threshold_point) for comp in candidate_spec.components}
    accepted = all(per.values())
    failing = tuple(c for c, ok in per.items() if not ok)
    return Decision(accepted, "supported" if accepted else "not_supported", failing, per)
