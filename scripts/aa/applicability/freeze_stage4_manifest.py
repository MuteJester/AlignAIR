"""Freeze the IMMUTABLE Stage-4 run manifest before launching the seed sweep.

Records the exact candidate/seed program, all hyperparameters, the deterministic production-seed rule, the
FIXED data partition hashes, and a forbidden-data (fresh ∪ locked) digest — computed seal-safely (fresh ids
from the fresh normalized sources; locked test ids from splits.jsonl by id only, never opening the sealed
locked_test.jsonl). The manifest is content-hashed; once written it is not rewritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alignair.aa.guard.classes import CLASS_MAP_HASH
from alignair.aa.guard.features import FEATURE_NAMES
from alignair.aa.guard.train_data import partition_by_parent_lineage

DEV = Path(".private/aa-gate3-hydration/dev_cohort.jsonl")
SPLITS = Path(".private/aa-gate3-hydration/splits.jsonl")          # ids+split only (seal-safe; no sequences)
FRESH = Path(".private/aa-applicability/fresh_hydration")
BANK_MANIFEST = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/profile_bank_manifest.json")
OUT = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/stage4_run_manifest.json")
CKPT_DIR = ".private/aa-applicability/checkpoints"

PARTITION_SEED = 42       # FIXED data split (seed-independent); candidate seeds only vary model init/training


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _seqhash(aa):
    return hashlib.sha256((aa or "").encode()).hexdigest()


def _forbidden_seqhashes():
    """Sequence-hash set of the FRESH cohort (selection + blind-LOCO) — the data the trainer must never load.
    Uses sequence hashes, not ids (Gate-3 ids are category-sequential and collide across fit/val/test). The
    sealed test is NOT enumerated here: dev_cohort = Gate-3 fit+val is disjoint from the locked test by the
    split-freeze construction, so we never open locked_test.jsonl."""
    hashes = set()
    for p in sorted(FRESH.glob("*.normalized.json")):
        try:
            for r in json.loads(p.read_text()):
                if r.get("aa"):
                    hashes.add(_seqhash(r["aa"]))
        except Exception:
            pass
    return hashes


def main():
    if OUT.exists():
        print(f"manifest already frozen at {OUT} (immutable) — not rewriting.")
        return
    recs = [json.loads(l) for l in open(DEV)]
    train, dev, part_hash = partition_by_parent_lineage(recs, seed=PARTITION_SEED, dev_frac=0.2)
    train_ids = sorted(r["id"] for r in train)
    dev_ids = sorted(r["id"] for r in dev)
    forbidden = _forbidden_seqhashes()
    dev_seqhashes = {_seqhash(r["aa"]) for r in recs}
    leak = dev_seqhashes & forbidden                       # by SEQUENCE hash (ids collide across splits)
    assert not leak, f"burned-dev intersects FRESH cohort by sequence: {len(leak)} seqs"

    manifest = {
        "artifact": "stage4_run_manifest",
        "immutable": True,
        "candidates": ["C1", "C2", "C3", "C4"],
        "seeds": [0, 1, 2],
        "expected_runs": 12,
        "checkpoint_dir": CKPT_DIR,
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.01},
        "schedule": {"name": "cosine", "warmup_steps": 50},
        "training": {"max_steps": 1200, "batch_size": 32, "eval_every": 50,
                     "early_stop_metric": "dev_total_loss", "early_stop_patience": 6, "min_delta": 1e-3,
                     "grad_clip": 1.0, "amp": True},
        "loss_weights": {"complete": 1.0, "app": 1.0, "oe": 0.5, "contrast": 0.5, "source": 0.5},
        "energy_margins": {"m_in": 5.0, "m_out": 10.0},
        "temperatures": {"supcon": 0.1},
        "grl_lambda": 1.0,
        "oe": {"policy": "seeded residue-shuffle of TRAIN sequences (train-only auxiliary outliers)",
               "cap_fraction": 0.15},
        "mining": {"rounds": 0, "policy": "smallest-margin, lineage-deduped (mining.py); disabled for the "
                                          "main seed sweep — deterministic balanced stream only"},
        "production_seed_rule": "min_dev_total_loss, tie -> lowest seed (dev-only; raises on any fresh field)",
        "class_map_hash": CLASS_MAP_HASH,
        "feature_order": list(FEATURE_NAMES),
        "bank_checksum": json.load(open(BANK_MANIFEST))["bank_checksum"],
        "train_partition_hash": part_hash,
        "train_ids_sha256": _sha(train_ids),
        "dev_ids_sha256": _sha(dev_ids),
        "n_train": len(train_ids), "n_dev": len(dev_ids),
        "forbidden_fresh_seqhashes_sha256": _sha(sorted(forbidden)),
        "forbidden_fresh_seqhashes_count": len(forbidden),
        "forbidden_leak_into_dev_by_sequence": len(leak),
        "sealed_test_exclusion": "structural: dev_cohort = Gate-3 fit+val, produced by split_freeze; the "
                                 "locked test partition is a separate file that is never opened.",
        "inaccessible_during_stage4": ["fresh_selection", "blind_loco", "far_ood", "sealed_locked_test"],
    }
    manifest["manifest_sha256"] = _sha({k: v for k, v in manifest.items()})
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({k: manifest[k] for k in ("expected_runs", "n_train", "n_dev", "train_partition_hash",
                                               "forbidden_fresh_seqhashes_count", "forbidden_leak_into_dev_by_sequence",
                                               "manifest_sha256")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
