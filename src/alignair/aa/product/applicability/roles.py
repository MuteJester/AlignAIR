"""Endpoint-specific fit/validation roles for Stage-2 (owner decision 2026-07-24).

Completeness and membership are DIFFERENT questions with DIFFERENT positives:

  * Completeness (is this a complete Ig/TCR V domain?) — positive-only fit on COMPLETE V DOMAINS of any
    locus: IGH (natural + engineered), IGK/IGL, TCR, murine VH, VHH, chimeric-variable. **IgSF is
    excluded** (its relationship to an immune-receptor V domain is too ambiguous to define the manifold).
    Negatives (validation only): the parent-aggregated structural transformations.

  * Membership (is this the supported human IGH VH manifold?) — positive-only fit on SUPPORTED COMPLETE
    HUMAN IGH only. Engineered/humanized are separately-reported CHALLENGE strata (not fit). Negatives
    (validation only): IGK/IGL, TCR, VHH/non-human, IgSF.

Roles are assigned only to NON-TEST (dev) records; the locked test is never touched. Fit/validation are
disjoint by lineage per endpoint; structural-parent lineages are kept OUT of both fit manifolds.
"""
from __future__ import annotations

import hashlib

FIT_FRAC = 0.60

_STRUCTURAL = {"reversed_vh", "v_only", "cdr3_only", "j_only", "leader_vh", "vh_constant"}
_IGH_NATURAL = {"pos_igh_repertoire", "pos_igh_simulator", "pos_igh_structure"}
_IGH_ENGINEERED = {"design_human_framework", "humanized"}
_COMPLETE_VDOMAIN = (_IGH_NATURAL | _IGH_ENGINEERED |
                     {"igk", "igl", "tra", "trb", "murine_vh", "vhh", "chimeric_murine_variable"})
_MEMBERSHIP_NEG = {"igk", "igl", "tra", "trb", "murine_vh", "vhh", "chimeric_murine_variable", "igsf_vset"}


def _u(lineage_key: str, salt: str) -> float:
    return int(hashlib.sha256((salt + "\x1f" + lineage_key).encode()).hexdigest()[:8], 16) / 0x100000000


def completeness_role(rec: dict, struct_parents: set) -> str | None:
    """`completeness_fit` | `completeness_validation` | None (excluded, e.g. IgSF)."""
    cat = rec["category"]
    if cat in _STRUCTURAL:
        return "completeness_validation"                       # negative (parent-aggregated at scoring)
    if cat == "igsf_vset":
        return None                                            # excluded from the completeness manifold
    if cat in _COMPLETE_VDOMAIN:
        if rec["lineage_key"] in struct_parents:               # a structural parent -> never in the fit manifold
            return "completeness_validation"
        return "completeness_fit" if _u(rec["lineage_key"], "compl") < FIT_FRAC else "completeness_validation"
    return None


def membership_role(rec: dict, struct_parents: set) -> str | None:
    """`membership_fit` | `membership_validation` | `membership_challenge` | None (not a membership record)."""
    cat = rec["category"]
    if cat in _MEMBERSHIP_NEG:
        return "membership_validation"                         # negative
    if cat in _IGH_ENGINEERED:
        return "membership_challenge"                          # reported separately, never fit
    if cat in _IGH_NATURAL:
        if rec["lineage_key"] in struct_parents:
            return "membership_validation"                     # positive but not in the fit manifold
        return "membership_fit" if _u(rec["lineage_key"], "memb") < FIT_FRAC else "membership_validation"
    return None                                                # structural: not a membership concern


def assign_roles(dev_records: list) -> dict:
    """Assign both endpoints' roles to the dev records + verify no fit/validation lineage overlap and no
    structural parent in either fit manifold. Returns {record_id: {completeness, membership}} + counts."""
    struct_parents = {r["lineage_key"] for r in dev_records if r["category"] in _STRUCTURAL}
    roles, counts = {}, {}
    fit_lin = {"completeness": set(), "membership": set()}
    val_lin = {"completeness": set(), "membership": set()}
    for r in dev_records:
        cr = completeness_role(r, struct_parents)
        mr = membership_role(r, struct_parents)
        roles[r["id"]] = {"completeness": cr, "membership": mr}
        for ep, role in (("completeness", cr), ("membership", mr)):
            if role and role.endswith("_fit"):
                fit_lin[ep].add(r["lineage_key"])
            elif role and role.endswith("_validation"):
                val_lin[ep].add(r["lineage_key"])
            counts[role] = counts.get(role, 0) + 1
    for ep in ("completeness", "membership"):
        assert not (fit_lin[ep] & val_lin[ep]), f"{ep} fit/validation lineage overlap"
    assert not (fit_lin["completeness"] & struct_parents), "structural parent in completeness fit"
    assert not (fit_lin["membership"] & struct_parents), "structural parent in membership fit"
    return {"roles": roles, "counts": counts}
