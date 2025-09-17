"""Model bundle schema definitions.

Defines dataclasses for versioned model bundle configuration and training metadata.
No heavy TensorFlow imports—keep this lightweight and pure-python for fast IO.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import sys
import platform
import json
from datetime import datetime

FORMAT_VERSION: int = 1


def utc_now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=None).isoformat() + "Z"


@dataclass
class ModelBundleConfig:
    """Schema describing the structural configuration of a saved model bundle.

    This intentionally captures only *structural* & reproducibility-critical fields.
    Anything that can be inferred from the underlying dataconfig at load-time is still
    captured redundantly here so we can validate mismatches early & explicitly.
    """
    model_type: str  # 'single_chain' | 'multi_chain'
    format_version: int
    max_seq_length: int
    has_d_gene: bool
    v_allele_count: int
    j_allele_count: int
    d_allele_count: Optional[int] = None
    v_allele_latent_size: Optional[int] = None
    j_allele_latent_size: Optional[int] = None
    d_allele_latent_size: Optional[int] = None
    chain_types: Optional[List[str]] = None  # multi-chain only
    number_of_chains: Optional[int] = None
    created_utc: str = utc_now_iso()
    alignairr_version: str = "unknown"
    python_version: str = platform.python_version()
    tf_version: str = "unknown"
    git_commit: Optional[str] = None
    notes: Optional[str] = None
    extra: Dict[str, Any] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelBundleConfig":
        return ModelBundleConfig(**d)


@dataclass
class TrainingMeta:
    """Metadata about the training run that produced the bundle.

    Non-architectural fields only. Architectural reproduction relies solely on ModelBundleConfig.
    """
    epochs_trained: int
    final_epoch: int
    best_epoch: Optional[int]
    best_loss: Optional[float]
    final_loss: Optional[float]
    metrics_summary: Dict[str, float]
    wall_time_seconds: Optional[int] = None
    batch_size: Optional[int] = None
    samples_per_epoch: Optional[int] = None
    optimizer_class: Optional[str] = None
    learning_rate: Optional[str] = None  # string to allow schedules
    mixed_precision: Optional[bool] = None
    created_utc: str = utc_now_iso()
    extra: Dict[str, Any] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainingMeta":
        return TrainingMeta(**d)


__all__ = [
    "FORMAT_VERSION",
    "ModelBundleConfig",
    "TrainingMeta",
]
