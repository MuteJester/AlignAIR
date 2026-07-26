"""Never-pooled selection / blind-LOCO allocator for the fresh validation cohort.

Each binding group is split into two DISJOINT subsets — a threshold-selection subset and a blind-LOCO
challenge subset — that are never pooled (spec 3.3/8.3). Splits are deterministic by sha256(lineage_key).

VHH is special: to extract the strongest possible unseen-category signal from a single source (SAbDab),
the preferred split is a GENUS holdout (selection = *L. glama*; blind-LOCO challenge = non-*L. glama*
camelids: vicugna / camelus). Deposition date is a soft ordering preference within the challenge (prefer
later depositions), NOT a hard filter — a hard temporal cutoff is unusable here because the pre-split
Gate-3 VHH span the latest depositions, leaving no post-cutoff window (Stage-1 empirical finding
2026-07-26; 537 *L. glama* / 501 non-*L. glama* seal-disjoint lineages available). When the data cannot
fill both genus sides to the VHH minimum (e.g. a single genus), the allocator falls back to a
deterministic sha-rank halving and RECORDS the mode, so a real-data fallback is a documented limitation
rather than a silent weakening.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

GUARDBAND_FLOOR = {"gate_5pct": 74, "gate_10pct": 36}

# Combined 5%-gate groups are balanced across their constituent loci so one locus cannot carry the gate.
# Per-locus floor = ceil(guardband_floor / #loci) = 37 for both (74 / 2). TRG/TRD stay descriptive.
COMBINED_LOCI = {"igk_igl": ["igk", "igl"], "tcr": ["tra", "trb"]}

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
    vhh_mode: str            # "genus_holdout" | "fallback_sha_rank" | "n/a"
    locus_counts: dict = field(default_factory=dict)   # combined group -> subset -> locus -> count
    locus_min_ok: dict = field(default_factory=dict)   # combined group -> subset -> bool (all loci >= min)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _rank(keys) -> list:
    return sorted(keys, key=_sha)


def _halve(lineages) -> tuple[set, set]:
    ordered = _rank(set(lineages))
    half = len(ordered) // 2
    return set(ordered[:half]), set(ordered[half:])


def _allocate_vhh_nonhuman(recs, cutoff) -> tuple[set, set, str]:
    """Return (selection_lineages, challenge_lineages, mode) for the vhh_nonhuman group.

    Genus holdout: selection = *L. glama*; challenge = non-*L. glama* camelids. `cutoff` is retained only as
    a soft ordering preference (later depositions preferred in the challenge); it never filters records out.
    """
    vhh = [r for r in recs if r.get("category") == "vhh"]
    other = [r for r in recs if r.get("category") != "vhh"]

    def _genus(r):
        return (r.get("heavy_species") or "").strip().lower()

    def _is_lama(r):
        return _genus(r) == _LAMA or "glama" in _genus(r)

    def _date(r):
        return r.get("deposition_date") or ""

    # Preferred: genus holdout (selection = L. glama; challenge = non-L. glama camelids).
    sel_vhh = {r["lineage_key"] for r in vhh if _is_lama(r)}
    # Soft temporal preference: order the challenge by later deposition first (does not filter).
    chal_recs = sorted((r for r in vhh if not _is_lama(r)), key=_date, reverse=True)
    chal_vhh = {r["lineage_key"] for r in chal_recs}

    if len(sel_vhh) >= VHH_MIN and len(chal_vhh) >= VHH_MIN and not (sel_vhh & chal_vhh):
        mode = "genus_holdout"
    else:
        # Fallback: deterministic sha-rank halving of ALL VHH (recorded; genus independence not achieved).
        sel_vhh, chal_vhh = _halve({r["lineage_key"] for r in vhh})
        mode = "fallback_sha_rank"

    # Non-VHH members (murine, chimeric) always split by sha-rank.
    sel_other, chal_other = _halve({r["lineage_key"] for r in other})
    return sel_vhh | sel_other, chal_vhh | chal_other, mode


def _allocate_combined(recs, loci, per_locus_min) -> tuple[set, set, dict, dict]:
    """Balanced split of a combined-locus group: each constituent category is halved independently, so
    both subsets carry a floor of every locus. Returns (selection, loco, counts, per-subset locus_ok)."""
    by_cat: dict = {}
    for r in recs:
        by_cat.setdefault(r["category"], []).append(r)
    sel: set = set()
    loc: set = set()
    counts: dict = {"selection": {}, "loco_challenge": {}}
    for cat, crecs in by_cat.items():
        s, l = _halve({r["lineage_key"] for r in crecs})
        sel |= s
        loc |= l
        counts["selection"][cat] = len(s)
        counts["loco_challenge"][cat] = len(l)
    ok = {sub: all(counts[sub].get(loc_, 0) >= per_locus_min for loc_ in loci)
          for sub in ("selection", "loco_challenge")}
    return sel, loc, counts, ok


def allocate_fresh(records, *, vhh_temporal_cutoff=None) -> Allocation:
    by_group: dict = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    selection: dict = {}
    loco: dict = {}
    locus_counts: dict = {}
    locus_min_ok: dict = {}
    vhh_mode = "n/a"
    for g, recs in by_group.items():
        if g == "vhh_nonhuman":
            sel, loc, vhh_mode = _allocate_vhh_nonhuman(recs, vhh_temporal_cutoff)
        elif g in COMBINED_LOCI:
            loci = COMBINED_LOCI[g]
            per = math.ceil(GUARDBAND_FLOOR[GROUP_GATE_KIND[g]] / len(loci))
            sel, loc, counts, ok = _allocate_combined(recs, loci, per)
            locus_counts[g] = counts
            locus_min_ok[g] = ok
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
                      vhh_min_ok=vhh_min_ok, pooled_overlap=pooled_overlap, vhh_mode=vhh_mode,
                      locus_counts=locus_counts, locus_min_ok=locus_min_ok)
