"""Stage-4 Task 4.2 — internal train/dev partitioning + balanced training stream.

Safeguard 5 (parent-lineage split): records are partitioned by PARENT lineage (``parent_lineage`` if present
else ``lineage_key``). Every structural transform of one parent — and the parent positive — lands in ONE
internal partition, so early stopping on the dev partition never sees a sibling transform of a training
parent. The partition assignment is deterministic (seeded) and hashed.

The training stream yields balanced batches of positives / known biological negatives / structural / OE
(outlier exposure). OE is a strict minority (fraction <= oe_cap) and can never dominate. No fresh or locked
id may appear — passing a forbidden set that intersects the stream RAISES.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict


def _parent_key(r):
    return r.get("parent_lineage") or r["lineage_key"]


def _role(r):
    g = r.get("group")
    if g in ("positive_real", "positive_sim"):
        return "positive"
    if g == "structural":
        return "structural"
    return "biological_negative"       # vhh_nonhuman, igsf, igk_igl, tcr, ...


def partition_by_parent_lineage(records, *, seed, dev_frac=0.2):
    """Deterministic parent-lineage partition -> (train, dev, partition_hash). Siblings never straddle."""
    groups = defaultdict(list)
    for r in records:
        groups[_parent_key(r)].append(r)
    ordered = sorted(groups, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    n_dev = round(len(ordered) * dev_frac)
    dev_keys = set(ordered[:n_dev])
    train, dev, id_part = [], [], {}
    for k in ordered:
        part = "dev" if k in dev_keys else "train"
        for r in groups[k]:
            (dev if part == "dev" else train).append(r)
            id_part[r["id"]] = part
    partition_hash = hashlib.sha256(json.dumps(id_part, sort_keys=True).encode()).hexdigest()
    return train, dev, partition_hash


def build_training_stream(train_records, oe_pool, *, seed, oe_cap=0.15, batch_size=32,
                          forbidden_ids=frozenset(), n_batches=None):
    """Yield balanced batches (positives / biological-neg / structural / OE-minority). Raises if any train
    or OE id is in `forbidden_ids` (fresh ∪ locked)."""
    forbidden = set(forbidden_ids)
    bad = [r["id"] for r in list(train_records) + list(oe_pool) if r.get("id") in forbidden]
    if bad:
        raise ValueError(f"forbidden (fresh/locked) ids present in training stream: {bad[:5]}")

    rng = random.Random(seed)
    by_role = defaultdict(list)
    for r in train_records:
        by_role[_role(r)].append(r)
    roles = [x for x in ("positive", "biological_negative", "structural") if by_role[x]]

    n_oe = int(batch_size * oe_cap) if oe_pool else 0        # OE strict minority: floor(batch*cap)/batch <= cap
    rest = batch_size - n_oe
    per_role = max(1, rest // len(roles))

    i = 0
    while n_batches is None or i < n_batches:
        batch = []
        for role in roles:
            pool = by_role[role]
            batch += [rng.choice(pool) for _ in range(per_role)]
        batch += [rng.choice(oe_pool) for _ in range(n_oe)]
        rng.shuffle(batch)
        yield batch
        i += 1
