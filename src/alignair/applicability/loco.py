"""Stage-5 — Leave-One-Category-Out (LOCO) fold machinery.

The corrected, biologically-COMPLETE group -> profile mask map (the plan's `mask vhh` example was stale):
each held-out cohort fold group masks ALL of its corresponding HMM-bank evidence profiles. The map is hashed
(LOCO_MASK_HASH) and must be bound into every LOCO checkpoint and result.

For a blind fold, the masked profiles are removed from: raw HMM feature calculation, competitive margins and
derived normalized features, the C0 rules and final decision graph, and reason attribution — no masked group
may appear in any reason.
"""
from __future__ import annotations

import hashlib
import json

# Cohort fold group -> HMM-bank evidence profile groups to mask (blind-LOCO).
LOCO_MASK_MAP = {
    "igk_igl": ("human_igk", "human_igl"),
    "tcr": ("tra", "trb", "trg", "trd"),
    "vhh_nonhuman": ("camelid_igh", "murine_vh"),
    "igsf": ("igsf_vset",),
}
LOCO_FOLDS = tuple(LOCO_MASK_MAP)              # the four held-out fold groups


def loco_mask_hash() -> str:
    return hashlib.sha256(json.dumps({k: sorted(v) for k, v in LOCO_MASK_MAP.items()},
                                     sort_keys=True).encode()).hexdigest()


LOCO_MASK_HASH = loco_mask_hash()


def masked_profiles_for(fold_g: str) -> frozenset:
    """The profile groups to mask when fold group `fold_g` is held out."""
    if fold_g not in LOCO_MASK_MAP:
        raise KeyError(f"unknown LOCO fold group: {fold_g!r} (folds: {LOCO_FOLDS})")
    return frozenset(LOCO_MASK_MAP[fold_g])


def loco_exclude(cohort, group_g: str):
    """Remove every record of fold group `group_g` (and any same-fold proxy category) from a cohort — used to
    build the LOCO training/threshold sets with the fold category fully withheld."""
    return [r for r in cohort if r.get("group") != group_g]


def assert_no_g_leak(train_ids, threshold_ids, g_ids):
    """Raise if any held-out fold id appears in the LOCO train or threshold sets."""
    g = set(g_ids)
    leak = (set(train_ids) | set(threshold_ids)) & g
    if leak:
        raise ValueError(f"LOCO leak: {len(leak)} fold ids in train/threshold sets: {sorted(leak)[:5]}")


def assert_reason_blind(reason: str, masked_groups) -> None:
    """Raise if a masked profile group name surfaces in a decision reason (blind-LOCO reason attribution)."""
    for g in masked_groups:
        if g in (reason or ""):
            raise ValueError(f"blind-LOCO reason leaks masked group {g!r}: {reason!r}")
