"""Development-only seed sweeps and leave-one-category-out training studies."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from alignair.aa.guard.loco import LOCO_FOLDS, LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.train_data import partition_by_parent_lineage
from alignair.errors import ArtifactIntegrityError
from .data import make_oe, precompute_features, source_setup
from .engine import RunConfig, resolve_device, train_one
from .reporting import write_loco_report, write_sweep_report
from .seeds import choose_production_seed

_LOG = logging.getLogger(__name__)


def _require_equal(actual, expected, message):
    if actual != expected:
        raise ArtifactIntegrityError(message)


def run_sweep(manifest, records, bank, *, artifact_dir, cache_path, device="cpu", checkpoint_dir=None):
    """Run the frozen candidate/seed grid and return runs plus chosen seeds.

    Inputs are supplied in memory. Output files contain diagnostics and hashes;
    this function does not discover or open validation or challenge datasets.
    """
    device = resolve_device(device)
    train, dev, partition_hash = partition_by_parent_lineage(records, seed=42, dev_frac=0.2)
    _require_equal(partition_hash, manifest["train_partition_hash"], "Partition drift versus training manifest.")
    _require_equal(bank.checksum(), manifest["bank_checksum"], "Profile bank differs from training manifest.")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(manifest, checkpoint_dir or manifest["checkpoint_dir"], artifact_dir / "budget.json", device)
    oe = make_oe(train, seed=42, cap_n=int(manifest["training"]["batch_size"] * manifest["oe"]["cap_fraction"]))
    cache = precompute_features(records + oe, bank, bank_checksum=manifest["bank_checksum"], cache_path=cache_path)
    cat2id, eligible, n_sources = source_setup(train)
    results = {}
    for cid in manifest["candidates"]:
        for seed in manifest["seeds"]:
            result = train_one(cid, seed, train, dev, oe, cache, cat2id, eligible, n_sources,
                               config=config, bank=bank)
            results[result["tag"]] = result
            _LOG.info("%s: %s (development loss %s)", result["tag"], result["status"], result.get("dev_total_loss"))
            (artifact_dir / "stage4_runs.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    frozen = {}
    for cid in manifest["candidates"]:
        losses = {r["seed"]: {"dev_loss": r["dev_total_loss"]} for r in results.values()
                  if r.get("status") == "ok" and r.get("cid") == cid}
        if losses:
            frozen[cid] = choose_production_seed(losses)
    write_sweep_report(results, frozen, artifact_dir=artifact_dir, manifest=manifest, device=device)
    return {"runs": results, "frozen_seeds": frozen}


def run_loco(training_manifest, manifest, records, bank, *, artifact_dir, cache_dir, device="cpu", checkpoint_dir=None):
    """Train each frozen candidate with one category withheld per fold.

    Fold membership restricts the original partition without reshuffling. Feature
    masking, normalization, and source eligibility are recomputed for each fold.
    """
    device = resolve_device(device)
    train, dev, partition_hash = partition_by_parent_lineage(records, seed=42, dev_frac=0.2)
    _require_equal(partition_hash, manifest["base_partition_hash"], "Partition drift versus LOCO manifest.")
    _require_equal(partition_hash, training_manifest["train_partition_hash"], "Partition drift versus training manifest.")
    _require_equal(bank.checksum(), training_manifest["bank_checksum"], "Profile bank differs from training manifest.")
    _require_equal(bank.checksum(), manifest["bank_checksum"], "Profile bank differs from LOCO manifest.")
    _require_equal(manifest["loco_mask_hash"], LOCO_MASK_HASH, "LOCO profile mask graph changed.")
    _require_equal(manifest["stage4_manifest_sha256"], training_manifest["manifest_sha256"],
                   "LOCO manifest refers to a different training manifest.")
    artifact_dir, cache_dir = Path(artifact_dir), Path(cache_dir)
    # Validate all fold bindings before launching any optimization.
    folds = {}
    for group in LOCO_FOLDS:
        fold_train = [r for r in train if r.get("group") != group]
        fold_dev = [r for r in dev if r.get("group") != group]
        cat2id, eligible, n_sources = source_setup(fold_train)
        frozen = manifest["folds"][group]
        _require_equal(sorted(cat2id), frozen["source_vocab_ordered"], f"Source vocabulary drift for {group}.")
        _require_equal(n_sources, frozen["n_sources"], f"Source count drift for {group}.")
        _require_equal(sorted(eligible), frozen["source_eligible_domain_types"], f"Source eligibility drift for {group}.")
        _require_equal(sorted(masked_profiles_for(group)), frozen["masked_profiles"], f"Profile mask drift for {group}.")
        excluded = [r for r in records if r.get("group") == group]
        bindings = {
            "loco_train_ids_sha256": sorted(r["id"] for r in fold_train),
            "loco_dev_ids_sha256": sorted(r["id"] for r in fold_dev),
            "excluded_ids_sha256": sorted(r["id"] for r in excluded),
            "excluded_seqhash_sha256": sorted(hashlib.sha256(r["aa"].encode()).hexdigest() for r in excluded),
        }
        for field, value in bindings.items():
            digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            _require_equal(digest, frozen[field], f"Record identity drift for {group}: {field}.")
        folds[group] = fold_train, fold_dev, cat2id, eligible, n_sources
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(training_manifest, checkpoint_dir or manifest["checkpoint_dir"], artifact_dir / "budget.json", device)
    results = {}
    for group, (fold_train, fold_dev, cat2id, eligible, n_sources) in folds.items():
        masked = masked_profiles_for(group)
        oe = make_oe(fold_train, seed=42, cap_n=int(training_manifest["training"]["batch_size"] *
                                                  training_manifest["oe"]["cap_fraction"]))
        cache = precompute_features(fold_train + fold_dev + oe, bank,
                                    bank_checksum=manifest["bank_checksum"], mask_groups=masked,
                                    cache_path=cache_dir / f"{group}.json")
        extra = {
            "loco_mask_hash": LOCO_MASK_HASH, "fold": group, "masked_profiles": sorted(masked),
            "partition_hash": manifest["folds"][group]["loco_train_ids_sha256"],
            "excluded_seqhash_sha256": manifest["folds"][group]["excluded_seqhash_sha256"],
            "training_manifest_hash": manifest["manifest_sha256"],
        }
        for cid in manifest["candidates"]:
            seed = manifest["frozen_seeds"][cid]
            tag = f"{cid}.s{seed}.{group}"
            started = time.time()
            result = train_one(cid, seed, fold_train, fold_dev, oe, cache, cat2id, eligible, n_sources,
                               config=config, bank=bank, mask_groups=masked, extra_provenance=extra, run_tag=tag)
            result["fold"] = group
            result.setdefault("wall_s", round(time.time() - started, 1))
            results[tag] = result
            _LOG.info("%s: %s (development loss %s)", tag, result["status"], result.get("dev_total_loss"))
            (artifact_dir / "loco_runs.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    write_loco_report(results, artifact_dir=artifact_dir, manifest=manifest, device=device)
    return results
