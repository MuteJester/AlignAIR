"""Training-study manifests: bind supplied settings to data and profile identities."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from alignair.aa.guard.classes import CLASS_MAP_HASH
from alignair.aa.guard.features import FEATURE_NAMES
from alignair.aa.guard.loco import LOCO_FOLDS, LOCO_MASK_HASH, masked_profiles_for
from alignair.aa.guard.train_data import partition_by_parent_lineage
from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError
from .seeds import choose_production_seed

# Settings can come from a previous study manifest; identities are always rebuilt.
SWEEP_SETTING_FIELDS = (
    "candidates", "seeds", "optimizer", "schedule", "training", "loss_weights",
    "energy_margins", "temperatures", "grl_lambda", "oe", "mining", "production_seed_rule",
)
PARTITION_SEED = 42


def _sha(value):
    # Preserve the existing training-manifest format, including JSON whitespace.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _settings(settings):
    if not isinstance(settings, dict) or any(key not in settings for key in SWEEP_SETTING_FIELDS):
        raise ConfigurationError("Sweep settings must supply every training setting.",
                                 context={"required_fields": SWEEP_SETTING_FIELDS})
    values = copy.deepcopy({key: settings[key] for key in SWEEP_SETTING_FIELDS})
    candidates, seeds = values["candidates"], values["seeds"]
    if (not isinstance(candidates, list) or not candidates or
            any(cid not in ("C1", "C2", "C3", "C4") for cid in candidates) or
            len(set(candidates)) != len(candidates)):
        raise ConfigurationError("Sweep candidates must be distinct learned guard candidates C1–C4.")
    if (not isinstance(seeds, list) or not seeds or
            any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ConfigurationError("Sweep seeds must be distinct integers.")
    for key in ("optimizer", "schedule", "training", "loss_weights", "energy_margins", "temperatures", "oe", "mining"):
        if not isinstance(values[key], dict):
            raise ConfigurationError("Training setting must be a mapping.", context={"setting": key})
    if values["production_seed_rule"] != "min_dev_total_loss, tie -> lowest seed (dev-only; raises on any fresh field)":
        raise ConfigurationError("Sweep settings must use the development-loss production-seed rule.")
    try:
        _sha(values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Sweep settings must contain finite JSON values.") from exc
    return values


def _records(records):
    records = list(records)
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or
                any(not isinstance(record.get(key), str) or not record[key] for key in ("id", "aa", "lineage_key")) or
                (record.get("parent_lineage") is not None and not isinstance(record["parent_lineage"], str))):
            raise DataContractError("Training records require text id, aa, and lineage fields.",
                                    context={"record_index": index})
    if len({record["id"] for record in records}) != len(records):
        raise DataContractError("Training record IDs must be unique when binding a partition.")
    return records


def build_sweep_manifest(records, *, settings, bank_checksum, checkpoint_dir, forbidden_sequence_hashes):
    """Bind a sweep to the fixed parent-lineage partition and supplied exclusions.

    No files are opened. Exclusion hashes must cover the fresh cohort; this check
    does not independently establish the origin of the supplied development data.
    Study settings are explicit, with no built-in historical experiment recipe.
    """
    values = _settings(settings)
    if not isinstance(bank_checksum, str) or not bank_checksum:
        raise DataContractError("A profile bank checksum is required.")
    records = _records(records)
    forbidden = set(forbidden_sequence_hashes)
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
           for value in forbidden):
        raise DataContractError("Fresh exclusion identities must be SHA-256 hexadecimal strings.")
    leak = {hashlib.sha256(record["aa"].encode()).hexdigest() for record in records} & forbidden
    if leak:
        raise ArtifactIntegrityError("Development records overlap the fresh cohort by sequence.",
                                     context={"overlap_count": len(leak)})
    train, dev, partition_hash = partition_by_parent_lineage(records, seed=PARTITION_SEED, dev_frac=0.2)
    if not train or not dev:
        raise ConfigurationError("A sweep requires nonempty training and development partitions.")
    manifest = {
        "artifact": "stage4_run_manifest", "immutable": True, **values,
        "expected_runs": len(values["candidates"]) * len(values["seeds"]),
        "checkpoint_dir": str(checkpoint_dir), "class_map_hash": CLASS_MAP_HASH,
        "feature_order": list(FEATURE_NAMES), "bank_checksum": bank_checksum,
        "train_partition_hash": partition_hash,
        "train_ids_sha256": _sha(sorted(record["id"] for record in train)),
        "dev_ids_sha256": _sha(sorted(record["id"] for record in dev)),
        "n_train": len(train), "n_dev": len(dev),
        "forbidden_fresh_seqhashes_sha256": _sha(sorted(forbidden)),
        "forbidden_fresh_seqhashes_count": len(forbidden),
        "forbidden_leak_into_dev_by_sequence": 0,
        "sealed_test_exclusion": "structural: dev_cohort = Gate-3 fit+val, produced by split_freeze; the "
                                 "locked test partition is a separate file that is never opened.",
        "inaccessible_during_stage4": ["fresh_selection", "blind_loco", "far_ood", "sealed_locked_test"],
    }
    manifest["manifest_sha256"] = _sha(manifest)
    return manifest


def freeze_sweep_manifest(records, *, output_path, settings, bank_checksum, checkpoint_dir,
                          forbidden_sequence_hashes):
    """Build and exclusively create a manifest; refuse to replace any existing file."""
    path = _new_output_path(output_path)
    manifest = build_sweep_manifest(records, settings=settings, bank_checksum=bank_checksum,
                                    checkpoint_dir=checkpoint_dir,
                                    forbidden_sequence_hashes=forbidden_sequence_hashes)
    return _write_manifest(path, manifest)


def _new_output_path(output_path):
    path = Path(output_path)
    if path.exists():
        raise ArtifactIntegrityError("A training manifest already exists.", context={"path": path})
    return path


def _write_manifest(path, manifest):
    serialized = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise ArtifactIntegrityError("Refusing to replace an existing training manifest.", context={"path": path}) from exc
    return manifest


def _production_seeds(settings, run_results):
    if not isinstance(run_results, dict) or any(not isinstance(result, dict) for result in run_results.values()):
        raise DataContractError("Sweep results must be a mapping of run records.")
    losses = {cid: {} for cid in settings["candidates"]}
    for result in run_results.values():
        if result.get("status") != "ok":
            continue
        cid, seed = result.get("cid"), result.get("seed")
        if (not isinstance(cid, str) or cid not in losses or isinstance(seed, bool) or
                not isinstance(seed, int) or seed not in settings["seeds"]):
            raise ArtifactIntegrityError("Successful sweep result is outside the frozen candidate/seed program.")
        if seed in losses[cid]:
            raise DataContractError("Duplicate successful candidate/seed result.", context={"candidate": cid, "seed": seed})
        # Preserve all fields so the shared selector can reject fresh-validation metrics.
        losses[cid][seed] = {**result, "dev_loss": result.get("dev_total_loss")}
    return {cid: choose_production_seed(results) for cid, results in losses.items()}


def build_loco_manifest(records, *, training_manifest, run_results, bank_checksum, checkpoint_dir,
                        input_files=()):
    """Restrict the frozen sweep partition for each LOCO fold, without reshuffling.

    Inputs are supplied in memory. No model, fresh cohort, or challenge files are
    opened. ``input_files`` records paths read by the caller for audit provenance.
    """
    settings = _settings(training_manifest)
    body = {key: value for key, value in training_manifest.items() if key != "manifest_sha256"}
    if training_manifest.get("manifest_sha256") != _sha(body):
        raise ArtifactIntegrityError("Training manifest hash does not reproduce.")
    for key, expected in (("bank_checksum", bank_checksum), ("class_map_hash", CLASS_MAP_HASH),
                          ("feature_order", list(FEATURE_NAMES))):
        if training_manifest.get(key) != expected:
            raise ArtifactIntegrityError("Training manifest identity differs from LOCO inputs.", context={"field": key})
    records = _records(records)
    if any(not isinstance(record.get(key), str) or not record[key]
           for record in records for key in ("group", "category")):
        raise DataContractError("LOCO records require text group and category fields.")
    train, dev, partition_hash = partition_by_parent_lineage(records, seed=PARTITION_SEED, dev_frac=0.2)
    identities = {"train_partition_hash": partition_hash,
                  "train_ids_sha256": _sha(sorted(record["id"] for record in train)),
                  "dev_ids_sha256": _sha(sorted(record["id"] for record in dev))}
    for key, expected in identities.items():
        if training_manifest.get(key) != expected:
            raise ArtifactIntegrityError("Development partition differs from the frozen sweep.", context={"field": key})
    seeds = _production_seeds(settings, run_results)
    from .data import source_setup

    folds = {}
    for group in LOCO_FOLDS:
        fold_train = [record for record in train if record["group"] != group]
        fold_dev = [record for record in dev if record["group"] != group]
        excluded = [record for record in records if record["group"] == group]
        if not fold_train or not fold_dev:
            raise ConfigurationError("LOCO fold requires nonempty training and development partitions.",
                                     context={"fold": group})
        vocabulary, eligible, count = source_setup(fold_train)
        ordered = sorted(vocabulary)
        folds[group] = {
            "masked_profiles": sorted(masked_profiles_for(group)),
            "n_train_after_exclusion": len(fold_train), "n_dev_after_exclusion": len(fold_dev),
            "n_excluded": len(excluded),
            "loco_train_ids_sha256": _sha(sorted(record["id"] for record in fold_train)),
            "loco_dev_ids_sha256": _sha(sorted(record["id"] for record in fold_dev)),
            "excluded_ids_sha256": _sha(sorted(record["id"] for record in excluded)),
            "excluded_seqhash_sha256": _sha(sorted(hashlib.sha256(record["aa"].encode()).hexdigest() for record in excluded)),
            "source_vocab_ordered": ordered, "source_vocab_sha256": _sha(ordered), "n_sources": count,
            "source_eligible_domain_types": sorted(eligible),
        }
    inherited = {key: settings[key] for key in SWEEP_SETTING_FIELDS if key not in (
        "candidates", "seeds", "production_seed_rule")}
    manifest = {
        "artifact": "loco_run_manifest", "immutable": True, "candidates": settings["candidates"],
        "frozen_seeds": seeds, "hyperparameters": "inherited unchanged from stage4_run_manifest.json",
        "stage4_manifest_sha256": training_manifest["manifest_sha256"], **inherited,
        "partition_policy": "RESTRICT the frozen Stage-4 parent-lineage partition after excluding fold g; "
                            "remaining records keep their original train/dev assignment (no reshuffle)",
        "base_partition_hash": partition_hash, "loco_mask_hash": LOCO_MASK_HASH,
        "bank_checksum": bank_checksum, "class_map_hash": CLASS_MAP_HASH, "checkpoint_dir": str(checkpoint_dir),
        "expected_models": len(settings["candidates"]) * len(LOCO_FOLDS), "folds": folds,
        "exclusion_scope": ["training", "internal_dev", "mining", "normalization_fit", "source_balancing", "early_stopping"],
        "files_opened_audit": {
            "opened": [str(path) for path in input_files],
            "NOT_opened": ["fresh_selection", "loco_challenge", "far_ood", "sealed_test"],
        },
    }
    manifest["manifest_sha256"] = _sha(manifest)
    return manifest


def freeze_loco_manifest(records, *, output_path, training_manifest, run_results, bank_checksum,
                         checkpoint_dir, input_files=()):
    """Exclusively create the LOCO manifest after validating its frozen sweep inputs."""
    path = _new_output_path(output_path)
    manifest = build_loco_manifest(records, training_manifest=training_manifest, run_results=run_results,
                                   bank_checksum=bank_checksum, checkpoint_dir=checkpoint_dir, input_files=input_files)
    return _write_manifest(path, manifest)
