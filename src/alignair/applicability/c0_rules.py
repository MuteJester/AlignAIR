"""Task 2.4 — C0 frozen HMM-only decision rules.

C0 is the profile-bank-only baseline (no encoder). Given `HmmFeatures`, it emits a `GuardVerdict` over the
four applicability states. The rule LADDER, score directions, and the finite threshold grid are FROZEN on
the burned Gate-3 development cohort before any fresh data is seen; the fresh selection subset later picks a
single predeclared grid point (Stage 6). No rule is invented or hand-tuned after viewing fresh data.

Ladder (first match wins):
  1. incomplete domain geometry (missing V or J, wrong order, or low combined coverage) -> insufficient_evidence
  2. human-IGH wins with sufficient score AND competitive margin -> supported
  3. a NAMED alternative locus wins by a validated margin AND is reason-precision-gated -> unsupported_locus:<locus>
  4. otherwise (weak/ambiguous, or a non-precision-gated winner) -> unknown_ood
"""
from __future__ import annotations

from dataclasses import dataclass

# Frozen threshold grid centre (burned-dev). A single grid point is chosen on fresh selection in Stage 6.
C0_THRESHOLDS = {
    "min_human_score": 20.0,      # bits: minimum human-IGH V bitscore to consider "supported"
    "min_human_margin": 5.0,      # bits: human-IGH must beat the best alternative locus by this
    "min_coverage": 0.80,         # combined V+J query coverage for a complete domain
    "min_alt_margin": 5.0,        # bits: a named alternative must beat human-IGH by this to claim its locus
    # reason-precision-gated alternative loci (allowlist established on burned-dev): only these may yield a
    # named unsupported_locus verdict; any other winner is unknown_ood.
    "precision_gated_loci": frozenset({"human_igk", "human_igl", "tra", "trb", "trg", "trd",
                                       "camelid_igh", "murine_vh", "igsf_vset"}),
}

STATES = ("supported", "insufficient_evidence", "unsupported_locus", "unknown_ood")


@dataclass(frozen=True)
class GuardVerdict:
    state: str        # one of STATES
    reason: str
    audit_score: float


def c0_decide(features, thresholds: dict = C0_THRESHOLDS) -> GuardVerdict:
    f = features
    # 1. complete-domain geometry gate (a V-only or mis-ordered/low-coverage hit is never a decision)
    if not (f.has_v and f.has_j and f.order_ok and f.combined_coverage >= thresholds["min_coverage"]):
        return GuardVerdict("insufficient_evidence", "incomplete_domain_geometry", round(f.combined_coverage, 4))

    # 2. human-IGH is the winning locus
    if f.best_v_group == "human_igh":
        if f.human_igh_score >= thresholds["min_human_score"] and f.margin >= thresholds["min_human_margin"]:
            return GuardVerdict("supported", "human_igh_supported", round(f.margin, 4))
        return GuardVerdict("unknown_ood", "human_igh_weak_margin", round(f.margin, 4))

    # 3. a named alternative locus wins; claim it only if the margin AND its reason-precision gate pass
    alt = f.best_v_group
    alt_margin = f.best_alt_score - f.human_igh_score          # how much the alternative beats human-IGH
    if alt_margin >= thresholds["min_alt_margin"] and alt in thresholds["precision_gated_loci"]:
        return GuardVerdict("unsupported_locus", f"alt_locus:{alt}", round(alt_margin, 4))

    # 4. weak / non-precision-gated winner
    return GuardVerdict("unknown_ood", f"alt_not_precision_gated:{alt}", round(alt_margin, 4))
