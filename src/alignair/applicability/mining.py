"""Stage-4 Task 4.3 — training-only hard-negative mining.

Selects the hardest auxiliary negatives (false-accepts / smallest decision margin), deduplicated by lineage.
It is TRAINING-ONLY: if any id in the auxiliary pool is in ``forbidden_ids`` (fresh ∪ locked), it RAISES —
mining must never draw from fresh selection, blind-LOCO, or the sealed test.
"""
from __future__ import annotations


def mine_hard_negatives(score_fn, aux_pool, *, forbidden_ids, top_k=64):
    """score_fn(record) -> margin (lower = harder / closer to a false accept). Returns the top_k hardest,
    lineage-deduped. Raises if aux_pool intersects forbidden_ids."""
    forbidden = set(forbidden_ids)
    intruders = [r["id"] for r in aux_pool if r.get("id") in forbidden]
    if intruders:
        raise ValueError(f"aux_pool intersects forbidden (fresh/locked) ids: {intruders[:5]}")

    best_by_lineage = {}
    for r in aux_pool:
        m = score_fn(r)
        lk = r.get("lineage_key", r.get("id"))
        if lk not in best_by_lineage or m < best_by_lineage[lk][0]:
            best_by_lineage[lk] = (m, r)
    ranked = [r for _, (m, r) in sorted(best_by_lineage.items(), key=lambda kv: kv[1][0])]
    return ranked[:top_k]
