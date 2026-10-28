"""Guard checkpoint identity checks and publication for trusted local training artifacts.

The training contract binds run inputs/settings. Provenance additionally checks runtime tokenizer,
profile bank, feature ordering, class map and partition identities. Other provenance fields remain
audit metadata. These checks do not authenticate pickle files or validate arbitrary model states.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

from alignair.errors import (
    ArtifactIntegrityError, ConfigurationError, ModelError, ModelFormatError, ModelIntegrityError,
    OperationIOError, error_context,
)
from alignair.io.atomic import atomic_write


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

    versions = {"torch": torch.__version__}
    for package in ("pyhmmer", "anarcii"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "n/a"
    return CheckpointProvenance(
        tokenizer=f"AaTokenizer(max_len={MAX_LEN},vocab={VOCAB_SIZE})",
        bank_checksum=bank_checksum, feature_order=tuple(FEATURE_NAMES), class_map_hash=CLASS_MAP_HASH,
        loss_mask_config=loss_mask_config, training_manifest_hash=training_manifest_hash,
        partition_hash=partition_hash, seed=seed,
        software_versions=versions)


# Identities that MUST match at load time (silent drift here = wrong features into wrong positions / wrong labels).
_BOUND = ("tokenizer", "bank_checksum", "feature_order", "class_map_hash", "partition_hash")


def verify_provenance(saved: dict, current: CheckpointProvenance):
    """Verify the five runtime identities; mismatch errors also remain catchable as ValueError."""
    with error_context(stage="guard_checkpoint"):
        cur = _current_identities(current)
        if not isinstance(saved, Mapping) or not _valid_identities(saved):
            raise ModelIntegrityError("Guard checkpoint provenance is missing or malformed.",
                                      context={"field": "provenance"},
                                      hint="Use a new run directory or explicitly migrate the checkpoint using verified runtime identities.")
        mismatch = [k for k in _BOUND if saved[k] != cur[k]]
        if mismatch:
            raise ModelIntegrityError(f"checkpoint provenance mismatch (cannot load): {mismatch}",
                                      context={"fields": mismatch})
    return True


def _current_identities(current):
    if not isinstance(current, CheckpointProvenance) or not _valid_identities(current.to_dict()):
        raise ConfigurationError("Guard checkpoint verification requires valid current identities.")
    return current.to_dict()


def _valid_identities(value):
    order = value.get("feature_order")
    return (all(isinstance(value.get(key), str) and value[key] for key in _BOUND if key != "feature_order")
            and isinstance(order, (list, tuple)) and bool(order)
            and all(isinstance(name, str) and name for name in order) and len(set(order)) == len(order))


def load_checkpoint(path, *, training_contract, provenance, device="cpu", sampling_plan=None):
    """Load a trusted local pickle and verify identities before the caller restores training state.

    Historical resume files without provenance require explicit migration or a new run. No
    fallback invents their runtime identities. Model/optimizer payload validation remains with
    the corresponding PyTorch state loaders.
    """
    import torch

    with error_context(stage="guard_checkpoint_load", path=str(path)):
        if not isinstance(training_contract, str) or not training_contract:
            raise ConfigurationError("Guard checkpoint verification requires a training contract.")
        _current_identities(provenance)
        try:
            expected_plan = json.dumps(sampling_plan, sort_keys=True, allow_nan=False) if sampling_plan is not None else None
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("Expected sampling plan must contain finite JSON values.") from exc
        try:
            saved = torch.load(path, map_location=device, weights_only=False)
        except OSError as exc:
            raise OperationIOError("Cannot read guard checkpoint.") from exc
        except (EOFError, pickle.UnpicklingError, ValueError) as exc:
            raise ModelFormatError("Cannot deserialize guard checkpoint.") from exc
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            raise ModelError("Cannot load guard checkpoint on the requested device.", context={"device": str(device)}) from exc
        if not isinstance(saved, Mapping):
            raise ModelFormatError("Guard checkpoint must contain a state mapping.")
        if not isinstance(saved.get("training_contract"), str) or saved["training_contract"] != training_contract:
            raise ArtifactIntegrityError(
                "Guard checkpoint does not match the requested training run.",
                hint="Use a new output directory for a different run. Legacy unbound checkpoints require explicit migration.",
            )
        verify_provenance(saved.get("provenance"), provenance)
        if expected_plan is not None:
            try:
                actual_plan = json.dumps(saved.get("sampling_plan"), sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ArtifactIntegrityError("Guard checkpoint sampling plan is malformed.",
                                             context={"field": "sampling_plan"}) from exc
            if actual_plan != expected_plan:
                raise ArtifactIntegrityError("Guard checkpoint sampling policy or allocation differs.",
                                             context={"field": "sampling_plan"})
        return saved


def save_checkpoint(path, state):
    """Atomically publish a guard state without replacing a completed predecessor on failure."""
    import torch

    with error_context(stage="guard_checkpoint_save", path=str(path)):
        try:
            atomic_write(path, lambda temporary: torch.save(state, temporary), stage="guard_checkpoint_save")
        except torch.cuda.OutOfMemoryError:
            raise
        except (OSError, RuntimeError) as exc:
            raise OperationIOError("Cannot publish guard checkpoint.") from exc
