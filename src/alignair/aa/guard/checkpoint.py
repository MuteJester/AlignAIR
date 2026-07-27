"""Stage-4 checkpoint provenance — every trained checkpoint BINDS the identities that make it reproducible
and safe to load: tokenizer, HMM bank checksum, feature order, class maps, loss/mask config, training-manifest
hash, partition hash, seed, and software versions. Loading verifies these and HARD-FAILS on any mismatch, so
a checkpoint can never be paired with a different bank / feature order / class map / tokenizer.

Safeguard 6 note: HMM normalization statistics are fit on the TRAIN partition only (and, later, per LOCO
fold) — never on the internal dev partition or the whole burned cohort. The bound ``partition_hash`` records
exactly which partition the stats belong to.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CheckpointProvenance:
    tokenizer: str
    bank_checksum: str
    feature_order: tuple
    class_map_hash: str
    loss_mask_config: dict
    training_manifest_hash: str
    partition_hash: str
    seed: int
    software_versions: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_order"] = list(self.feature_order)
        return d

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def build_provenance(*, bank_checksum, training_manifest_hash, partition_hash, seed, loss_mask_config):
    from alignair.aa.guard.classes import CLASS_MAP_HASH
    from alignair.aa.guard.features import FEATURE_NAMES
    from alignair.aa.guard.tokenizer import MAX_LEN, VOCAB_SIZE
    import torch

    import pyhmmer
    try:
        import anarcii
        av = getattr(anarcii, "__version__", "installed")
    except Exception:  # noqa: BLE001
        av = "n/a"
    return CheckpointProvenance(
        tokenizer=f"AaTokenizer(max_len={MAX_LEN},vocab={VOCAB_SIZE})",
        bank_checksum=bank_checksum, feature_order=tuple(FEATURE_NAMES), class_map_hash=CLASS_MAP_HASH,
        loss_mask_config=loss_mask_config, training_manifest_hash=training_manifest_hash,
        partition_hash=partition_hash, seed=seed,
        software_versions={"torch": torch.__version__, "pyhmmer": pyhmmer.__version__, "anarcii": av})


# Identities that MUST match at load time (silent drift here = wrong features into wrong positions / wrong labels).
_BOUND = ("tokenizer", "bank_checksum", "feature_order", "class_map_hash", "partition_hash")


def verify_provenance(saved: dict, current: CheckpointProvenance):
    """Raise if any bound identity differs between a saved checkpoint and the current environment."""
    cur = current.to_dict()
    mismatch = [k for k in _BOUND if saved.get(k) != cur.get(k)]
    if mismatch:
        raise ValueError(f"checkpoint provenance mismatch (cannot load): {mismatch}")
    return True
