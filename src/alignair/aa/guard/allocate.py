"""Never-pooled selection / blind-LOCO allocator for the fresh validation cohort.

Each binding group is split into two DISJOINT subsets — a threshold-selection subset and a blind-LOCO
challenge subset — that are never pooled (spec 3.3/8.3). Splits are deterministic by sha256(lineage_key).

VHH is special: to extract the strongest possible unseen-category signal from a single source (SAbDab),
the preferred split is a genus + temporal-deposition holdout (selection = *L. glama* at/before the frozen
cutoff; challenge = non-*L. glama* after it). When the data cannot fill both sides to the VHH minimum
(e.g. a single genus), the allocator falls back to a deterministic sha-rank halving and RECORDS the mode,
so Stage-1 execution can treat a real-data fallback as a documented limitation (genus independence not
achieved) rather than a silent weakening.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

GUARDBAND_FLOOR = {"gate_5pct": 74, "gate_10pct": 36}

# Which guardband floor each binding group is held to (5% gates vs 10% gates).
GROUP_GATE_KIND = {
    "igk_igl": "gate_5pct",
    "tcr": "gate_5pct",
    "vhh_nonhuman": "gate_10pct",
    "igsf": "gate_10pct",
    "structural": "gate_10pct",
    "positive_real": "gate_10pct",
    "positive_sim": "gate_10pct",
}

VHH_MIN = 19
_LAMA = "lama glama"


@dataclass(frozen=True)
class Allocation:
    selection: dict          # group -> set[lineage_key]
    loco_challenge: dict     # group -> set[lineage_key]
    floors_ok: dict          # group -> bool (both subsets meet the guardband floor)
    vhh_min_ok: dict         # {"selection": bool, "loco_challenge": bool} (>= VHH_MIN VHH each)
    pooled_overlap: set      # must be empty
    vhh_mode: str            # "genus_temporal_holdout" | "fallback_sha_rank" | "n/a"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _rank(keys) -> list:
    return sorted(keys, key=_sha)


def _halve(lineages) -> tuple[set, set]:
    ordered = _rank(set(lineages))
    half = len(ordered) // 2
    return set(ordered[:half]), set(ordered[half:])


def _allocate_vhh_nonhuman(recs, cutoff) -> tuple[set, set, str]:
    """Return (selection_lineages, challenge_lineages, mode) for the vhh_nonhuman group."""
    vhh = [r for r in recs if r.get("category") == "vhh"]
    other = [r for r in recs if r.get("category") != "vhh"]

    def _genus(r):
        return (r.get("heavy_species") or "").strip().lower()

    def _date(r):
        return r.get("deposition_date") or ""

    # Preferred: strict genus + temporal holdout.
    sel_vhh = {r["lineage_key"] for r in vhh
               if _genus(r) == _LAMA and (cutoff is None or _date(r) <= cutoff)}
    chal_vhh = {r["lineage_key"] for r in vhh
                if _genus(r) != _LAMA and (cutoff is None or _date(r) > cutoff)}

    if len(sel_vhh) >= VHH_MIN and len(chal_vhh) >= VHH_MIN and not (sel_vhh & chal_vhh):
        mode = "genus_temporal_holdout"
    else:
        # Fallback: deterministic sha-rank halving of ALL VHH (recorded; genus independence not achieved).
        sel_vhh, chal_vhh = _halve({r["lineage_key"] for r in vhh})
        mode = "fallback_sha_rank"

    # Non-VHH members (murine, chimeric) always split by sha-rank.
    sel_other, chal_other = _halve({r["lineage_key"] for r in other})
    return sel_vhh | sel_other, chal_vhh | chal_other, mode


def allocate_fresh(records, *, vhh_temporal_cutoff=None) -> Allocation:
    by_group: dict = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    selection: dict = {}
    loco: dict = {}
    vhh_mode = "n/a"
    for g, recs in by_group.items():
        if g == "vhh_nonhuman":
            sel, loc, vhh_mode = _allocate_vhh_nonhuman(recs, vhh_temporal_cutoff)
        else:
            sel, loc = _halve({r["lineage_key"] for r in recs})
        selection[g] = sel
        loco[g] = loc

    floors_ok: dict = {}
    for g in by_group:
        floor = GUARDBAND_FLOOR[GROUP_GATE_KIND.get(g, "gate_10pct")]
        floors_ok[g] = len(selection[g]) >= floor and len(loco[g]) >= floor

    vhh_lineages = {r["lineage_key"] for r in by_group.get("vhh_nonhuman", [])
                    if r.get("category") == "vhh"}
    vhh_min_ok = {
        "selection": len(selection.get("vhh_nonhuman", set()) & vhh_lineages) >= VHH_MIN,
        "loco_challenge": len(loco.get("vhh_nonhuman", set()) & vhh_lineages) >= VHH_MIN,
    }

    pooled_overlap: set = set()
    for g in by_group:
        pooled_overlap |= (selection[g] & loco[g])

    return Allocation(selection=selection, loco_challenge=loco, floors_ok=floors_ok,
                      vhh_min_ok=vhh_min_ok, pooled_overlap=pooled_overlap, vhh_mode=vhh_mode)
