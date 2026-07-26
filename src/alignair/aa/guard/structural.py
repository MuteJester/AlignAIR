"""Structural negative generation from FINALIZED real-IGH parents.

Generated LAST, only from REAL IGH parents (never simulator) so the completeness gate is not
simulator-specific. Each of the six deterministic transforms inherits its parent's lineage and role
(parent-level scoring: a parent is a false-accept if ANY of its transforms is accepted). Expected reasons
map to the completeness taxonomy.
"""
from __future__ import annotations

# Representative human IGH signal peptide + IGHG1 CH1 N-terminal fragment (fixed contamination sequences).
LEADER = "MELGLSWIFLLAILKGVQC"
CH1 = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEP"

# category -> (transformation id, expected reason)
TRANSFORMS = {
    "reversed_vh": ("structural_reverse_v1", "reversed_order"),
    "v_only": ("structural_v_region_v1", "missing_j_anchor"),
    "j_only": ("structural_j_region_v1", "missing_v_anchor"),
    "cdr3_only": ("structural_cdr3_only_v1", "too_short"),
    "leader_vh": ("structural_leader_prepend_v1", "leader_or_constant_contamination"),
    "vh_constant": ("structural_ch1_append_v1", "leader_or_constant_contamination"),
}


def make_structural(parent_aa: str, cdr3_aa: str) -> dict:
    """Deterministic six-transform dict for one real-IGH parent VH sequence."""
    idx = parent_aa.find(cdr3_aa) if cdr3_aa else -1
    if idx >= 0:
        v_only = parent_aa[:idx]                      # V region up to CDR3
        j_only = parent_aa[idx:]                      # CDR3 -> FR4 (missing V anchor)
        cdr3_only = cdr3_aa
    else:
        cut = len(parent_aa) * 2 // 3
        v_only = parent_aa[:cut]
        j_only = parent_aa[cut:]
        cdr3_only = parent_aa[len(parent_aa) // 2: len(parent_aa) // 2 + 10]
    return {
        "reversed_vh": parent_aa[::-1],
        "v_only": v_only,
        "j_only": j_only,
        "cdr3_only": cdr3_only,
        "leader_vh": LEADER + parent_aa,
        "vh_constant": parent_aa + CH1,
    }


def structural_records(parent: dict, *, role_of_parent: str) -> list:
    """Structural records for a REAL-IGH parent, inheriting its lineage + role.

    Raises ValueError if the parent is not a real IGH positive (never derive structural from simulator).
    """
    if parent.get("group") != "positive_real":
        raise ValueError(f"structural parents must be real IGH (positive_real); got {parent.get('group')!r}")
    transforms = make_structural(parent["aa"], parent.get("cdr3_aa", ""))
    out = []
    for cat, aa in transforms.items():
        tv, reason = TRANSFORMS[cat]
        out.append({
            "id": f"{cat}_{parent['lineage_key']}",
            "aa": aa,
            "group": "structural",
            "category": cat,
            "lineage_key": parent["lineage_key"],       # inherit parent lineage (parent-level scoring)
            "parent_lineage": parent["lineage_key"],
            "expected_state": "insufficient_evidence",
            "expected_reason": reason,
            "transformation": tv,
            "inherited_role": role_of_parent,
            "source": "structural",
        })
    return out
