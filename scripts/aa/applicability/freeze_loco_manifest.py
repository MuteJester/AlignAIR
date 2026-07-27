"""Freeze the IMMUTABLE LOCO run manifest before training the 16 fold models.

Enforces the owner's constraints: C1-C4 keep their FROZEN Stage-4 seeds + unchanged hyperparameters; the
existing parent-lineage train/dev partition is RESTRICTED (records of fold group g removed, remaining records
keep their original assignment — NO reshuffle); g is excluded from train/dev/mining/normalization/source-
balancing/early-stop; normalization is fit per fold on the remaining train; source-adversarial eligibility +
the ORDERED source vocabulary are recomputed per fold and hashed; each fold binds LOCO_MASK_HASH, bank
checksum, class-map hash, partition hashes, and an excluded-record digest. No fresh/LOCO-challenge/far-OOD/
sealed file is opened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alignair.aa.guard.classes import CLASS_MAP_HASH
from alignair.aa.guard.loco import LOCO_FOLDS, LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.train_data import partition_by_parent_lineage

ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")
STAGE4_MAN = json.load(open(ART / "stage4_run_manifest.json"))
STAGE4_RUNS = json.load(open(ART / "stage4_runs.json"))
DEV = Path(".private/aa-gate3-hydration/dev_cohort.jsonl")
BANK_MANIFEST = json.load(open(ART / "profile_bank_manifest.json"))
OUT = ART / "loco_run_manifest.json"
LOCO_CKPT = ".private/aa-applicability/loco_checkpoints"


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _frozen_seeds():
    seeds = {}
    for cid in STAGE4_MAN["candidates"]:
        dev = {r["seed"]: r["dev_total_loss"] for r in STAGE4_RUNS.values()
               if r.get("status") == "ok" and r.get("cid") == cid}
        seeds[cid] = min(dev, key=lambda s: (dev[s], s))       # committed dev-only rule
    return seeds


def _source_vocab(train_recs):
    from collections import defaultdict
    import sys
    sys.path.insert(0, str(Path("scripts/aa/applicability").resolve()))
    from run_stage4_sweep import _domain_type
    cats_by_dt = defaultdict(set)
    for r in train_recs:
        dt = _domain_type(r)
        if dt >= 0:
            cats_by_dt[dt].add(r.get("category", ""))
    all_cats = sorted({c for cats in cats_by_dt.values() for c in cats})   # ORDERED source vocabulary
    eligible = sorted(dt for dt, cats in cats_by_dt.items() if len(cats) >= 2)
    return all_cats, eligible


def main():
    if OUT.exists():
        print(f"LOCO manifest already frozen at {OUT} (immutable) — not rewriting.")
        return
    recs = [json.loads(l) for l in open(DEV)]
    train, dev, part_hash = partition_by_parent_lineage(recs, seed=42, dev_frac=0.2)
    assert part_hash == STAGE4_MAN["train_partition_hash"], "partition drift vs Stage-4"

    folds = {}
    for g in LOCO_FOLDS:
        loco_train = [r for r in train if r.get("group") != g]     # RESTRICT existing partition (no reshuffle)
        loco_dev = [r for r in dev if r.get("group") != g]
        excluded = [r for r in recs if r.get("group") == g]
        vocab, eligible = _source_vocab(loco_train)
        folds[g] = {
            "masked_profiles": sorted(masked_profiles_for(g)),
            "n_train_after_exclusion": len(loco_train),
            "n_dev_after_exclusion": len(loco_dev),
            "n_excluded": len(excluded),
            "loco_train_ids_sha256": _sha(sorted(r["id"] for r in loco_train)),
            "loco_dev_ids_sha256": _sha(sorted(r["id"] for r in loco_dev)),
            "excluded_ids_sha256": _sha(sorted(r["id"] for r in excluded)),
            "excluded_seqhash_sha256": _sha(sorted(hashlib.sha256(r["aa"].encode()).hexdigest() for r in excluded)),
            "source_vocab_ordered": vocab,
            "source_vocab_sha256": _sha(vocab),
            "n_sources": len(vocab),
            "source_eligible_domain_types": eligible,
        }

    manifest = {
        "artifact": "loco_run_manifest", "immutable": True,
        "candidates": STAGE4_MAN["candidates"],
        "frozen_seeds": _frozen_seeds(),
        "hyperparameters": "inherited unchanged from stage4_run_manifest.json",
        "stage4_manifest_sha256": STAGE4_MAN["manifest_sha256"],
        "optimizer": STAGE4_MAN["optimizer"], "schedule": STAGE4_MAN["schedule"],
        "training": STAGE4_MAN["training"], "loss_weights": STAGE4_MAN["loss_weights"],
        "energy_margins": STAGE4_MAN["energy_margins"], "temperatures": STAGE4_MAN["temperatures"],
        "grl_lambda": STAGE4_MAN["grl_lambda"], "oe": STAGE4_MAN["oe"], "mining": STAGE4_MAN["mining"],
        "partition_policy": "RESTRICT the frozen Stage-4 parent-lineage partition after excluding fold g; "
                            "remaining records keep their original train/dev assignment (no reshuffle)",
        "base_partition_hash": part_hash,
        "loco_mask_hash": LOCO_MASK_HASH,
        "bank_checksum": BANK_MANIFEST["bank_checksum"],
        "class_map_hash": CLASS_MAP_HASH,
        "checkpoint_dir": LOCO_CKPT,
        "expected_models": len(STAGE4_MAN["candidates"]) * len(LOCO_FOLDS),
        "folds": folds,
        "exclusion_scope": ["training", "internal_dev", "mining", "normalization_fit", "source_balancing",
                            "early_stopping"],
        "files_opened_audit": {
            "opened": [str(DEV), ".private/aa-applicability/profiles/*", "stage4 manifests (config)"],
            "NOT_opened": ["fresh_selection (*.normalized.json fresh cohort)", "loco_challenge subsets",
                           "far_ood pools", ".private/aa-gate3-hydration/locked_test.jsonl (SEAL)"],
        },
    }
    manifest["manifest_sha256"] = _sha({k: v for k, v in manifest.items()})
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"expected_models": manifest["expected_models"], "frozen_seeds": manifest["frozen_seeds"],
                      "folds": {g: {"n_train": folds[g]["n_train_after_exclusion"],
                                    "n_excluded": folds[g]["n_excluded"], "n_sources": folds[g]["n_sources"]}
                                for g in folds},
                      "manifest_sha256": manifest["manifest_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
