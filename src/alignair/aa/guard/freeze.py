"""Stage-5 Task 5.3 — freeze EVERY guard identity together into one immutable pre-unlock bundle.

Before the first fresh-selection read, the owner requires these to be frozen as a unit: model, normalization,
source vocabulary, class map, profile bank, feature ordering, mask graph, preprocessing, threshold grid,
evaluator commit, and cohort-manifest. This module hashes each and folds them into a single ``bundle_sha256``.
The bundle is the authoritative ``expected_hashes`` the challenge loader later checks — any drift hard-fails.

No sequences, no model weights are stored here — hashes/counts only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from alignair.aa.guard.classes import CLASS_MAP_HASH
from alignair.aa.guard.features import FEATURE_NAMES
from alignair.aa.guard.loco import LOCO_MASK_HASH
from alignair.aa.guard.tokenizer import MAX_LEN, VOCAB_SIZE

# The frozen slots every pre-unlock bundle MUST carry (owner's enumerated list). A missing slot is a defect.
REQUIRED_SLOTS = (
    "model", "normalization", "source_vocab", "class_map", "profile_bank", "feature_order",
    "mask_graph", "preprocessing", "threshold_grid", "evaluator_commit", "cohort_manifest",
)


def _sha_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_order_hash() -> str:
    """Hash of the exact ordered feature-name list fusion positions depend on."""
    return _sha_json(list(FEATURE_NAMES))


def model_identity(provenance_digest: str, checkpoint_file_sha256: str) -> str:
    """The ``model`` slot binds BOTH the provenance metadata digest AND the actual checkpoint-file bytes, so a
    weight file swapped under identical metadata drifts this hash and hard-fails at challenge unlock."""
    return _sha_json({"provenance_digest": provenance_digest, "checkpoint_file_sha256": checkpoint_file_sha256})


def preprocessing_hash() -> str:
    """Tokenizer + length-contract identity (pads-only, no-truncation contract)."""
    return _sha_json({"tokenizer": f"AaTokenizer(max_len={MAX_LEN},vocab={VOCAB_SIZE})",
                      "length_contract_max_len": MAX_LEN, "truncation": "forbidden"})


@dataclass(frozen=True)
class FrozenBundle:
    hashes: dict            # slot -> sha256 (every REQUIRED_SLOT present)

    def __post_init__(self):
        missing = [s for s in REQUIRED_SLOTS if s not in self.hashes]
        if missing:
            raise ValueError(f"pre-unlock bundle missing frozen slots: {missing}")

    def bundle_sha256(self) -> str:
        return _sha_json(self.hashes)

    def to_dict(self) -> dict:
        return {"hashes": dict(self.hashes), "bundle_sha256": self.bundle_sha256(),
                "required_slots": list(REQUIRED_SLOTS)}


def freeze_pre_unlock_bundle(*, model_hash, normalization_hash, source_vocab_hash, bank_checksum,
                             evaluator_commit, cohort_manifest_hash, threshold_grid_hash) -> FrozenBundle:
    """Fold every guard identity into one bundle. The self-derivable slots (class map, feature order, mask
    graph, preprocessing) are pinned from source here; the rest are passed in from the frozen artifacts."""
    return FrozenBundle(hashes={
        "model": model_hash,                       # fold checkpoint provenance digest (or synthetic, in the mock)
        "normalization": normalization_hash,       # per-fold HMM norm-stat hash
        "source_vocab": source_vocab_hash,         # ordered source vocabulary hash
        "class_map": CLASS_MAP_HASH,               # domain-type / completeness class semantics
        "profile_bank": bank_checksum,             # competitive HMM bank checksum
        "feature_order": feature_order_hash(),     # exact fusion feature ordering
        "mask_graph": LOCO_MASK_HASH,              # LOCO group->profile mask map
        "preprocessing": preprocessing_hash(),     # tokenizer + no-truncation length contract
        "threshold_grid": threshold_grid_hash,     # predeclared finite operating-point grid + directions + gate
        "evaluator_commit": evaluator_commit,      # git HEAD the evaluation code was frozen at
        "cohort_manifest": cohort_manifest_hash,   # FROZEN_COHORT_MANIFEST identity
    })
