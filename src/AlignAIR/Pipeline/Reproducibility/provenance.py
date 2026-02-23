"""Run provenance — complete record of what produced a pipeline result."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from AlignAIR.Pipeline.Reproducibility.environment import EnvironmentFingerprint


@dataclass
class RunProvenance:
    """Complete provenance for one pipeline run.

    If two runs share the same ``reproducibility_hash`` AND determinism
    was enabled, outputs should be bit-identical.
    """

    run_id: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    wall_seconds: float = 0.0

    # Environment
    environment: Optional[Dict[str, str]] = None

    # Inputs
    input_path: str = ""
    input_sha256: str = ""
    n_sequences: int = 0

    # Model
    model_dir: str = ""
    model_fingerprint: str = ""

    # Config
    config_sha256: str = ""

    # Determinism
    determinism_settings: Optional[Dict[str, str]] = None

    # Stages
    stages_executed: List[str] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)

    # Output
    output_path: str = ""
    output_sha256: str = ""

    # Composite
    reproducibility_hash: str = ""

    def compute_reproducibility_hash(self) -> str:
        """SHA-256 of the inputs that fully determine the output."""
        parts = "|".join([
            self.input_sha256,
            self.model_fingerprint,
            self.config_sha256,
            self.environment.get("fingerprint", "") if self.environment else "",
        ])
        self.reproducibility_hash = hashlib.sha256(parts.encode()).hexdigest()[:16]
        return self.reproducibility_hash

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> RunProvenance:
        data = json.loads(text)
        return cls(**data)


def file_sha256(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
