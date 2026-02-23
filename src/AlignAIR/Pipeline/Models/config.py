from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from AlignAIR.Pipeline.Models.enums import OutputFormat, ThresholdMethod


@dataclass(frozen=True)
class AlleleThresholdConfig:
    """Per-gene allele thresholding parameters."""
    v_threshold: float = 0.1
    d_threshold: float = 0.1
    j_threshold: float = 0.1
    v_cap: int = 3
    d_cap: int = 3
    j_cap: int = 3
    method: ThresholdMethod = ThresholdMethod.MAX_LIKELIHOOD_PERCENTAGE


@dataclass(frozen=True)
class OrientationConfig:
    """Sequence orientation correction settings."""
    enabled: bool = True
    custom_model_path: Optional[str] = None


@dataclass(frozen=True)
class MemoryConfig:
    """Memory management tuning."""
    batch_size: int = 2048
    mmap_threshold_gb: float = 2.0
    auto_tune_batch: bool = False
    max_memory_fraction: float = 0.6


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint/resume settings."""
    enabled: bool = False
    directory: Optional[str] = None


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Determinism and reproducibility settings."""
    seed: int = 42
    deterministic_ops: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Single source of truth for an entire pipeline run.

    Frozen after construction — never mutated by stages.
    Constructed from CLI args, YAML file, or programmatic API.
    Serializable to JSON for reproducibility.
    """
    # Required paths
    model_dir: str = ""
    sequences_path: str = ""
    save_path: str = ""

    # Output
    output_format: OutputFormat = OutputFormat.CSV
    translate_to_asc: bool = True

    # Genotype
    custom_genotype_path: Optional[str] = None

    # Sub-configs
    thresholds: AlleleThresholdConfig = field(default_factory=AlleleThresholdConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    def to_dict(self) -> dict:
        """Convert to a plain dict (enums become their values)."""
        d = asdict(self)
        d["output_format"] = self.output_format.value
        d["thresholds"]["method"] = self.thresholds.method.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def sha256(self) -> str:
        """Deterministic hash for provenance."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, raw: dict) -> PipelineConfig:
        """Construct from a plain dict (e.g., parsed YAML or JSON)."""
        # Handle nested dataclasses
        if "thresholds" in raw and isinstance(raw["thresholds"], dict):
            t = raw["thresholds"]
            if "method" in t and isinstance(t["method"], str):
                t["method"] = ThresholdMethod(t["method"])
            raw["thresholds"] = AlleleThresholdConfig(**t)
        if "orientation" in raw and isinstance(raw["orientation"], dict):
            raw["orientation"] = OrientationConfig(**raw["orientation"])
        if "memory" in raw and isinstance(raw["memory"], dict):
            raw["memory"] = MemoryConfig(**raw["memory"])
        if "checkpoint" in raw and isinstance(raw["checkpoint"], dict):
            raw["checkpoint"] = CheckpointConfig(**raw["checkpoint"])
        if "reproducibility" in raw and isinstance(raw["reproducibility"], dict):
            raw["reproducibility"] = ReproducibilityConfig(**raw["reproducibility"])
        # Handle enum
        if "output_format" in raw and isinstance(raw["output_format"], str):
            raw["output_format"] = OutputFormat(raw["output_format"])
        # Ignore unknown keys for forward compatibility
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, text: str) -> PipelineConfig:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
