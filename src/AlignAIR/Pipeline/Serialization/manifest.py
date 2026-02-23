"""Pipeline manifest — schema-versioned record of all artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactEntry:
    """One file in the pipeline archive."""

    path: str                  # Relative path within archive
    format: str                # 'npz' | 'json' | 'csv' | 'tsv'
    sha256: str                # Hex digest
    size_bytes: int
    description: str = ""
    array_shapes: Optional[Dict[str, List[int]]] = None


@dataclass
class PipelineManifest:
    """Schema-versioned manifest of a pipeline run's artifacts."""

    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    created_utc: str = ""
    alignair_version: str = ""
    n_sequences: int = 0
    stages_completed: List[str] = field(default_factory=list)
    artifacts: List[ArtifactEntry] = field(default_factory=list)

    def add_artifact(
        self,
        archive_dir: Path,
        relative_path: str,
        fmt: str,
        description: str = "",
    ) -> ArtifactEntry:
        """Register a file as an artifact, computing its SHA-256."""
        full_path = archive_dir / relative_path
        sha = _file_sha256(full_path)
        size = full_path.stat().st_size
        entry = ArtifactEntry(
            path=relative_path,
            format=fmt,
            sha256=sha,
            size_bytes=size,
            description=description,
        )
        self.artifacts.append(entry)
        return entry

    def validate_integrity(self, archive_dir: Path) -> List[str]:
        """Check SHA-256 of every artifact. Returns list of failures."""
        failures = []
        for art in self.artifacts:
            full_path = archive_dir / art.path
            if not full_path.exists():
                failures.append(f"MISSING: {art.path}")
                continue
            actual = _file_sha256(full_path)
            if actual != art.sha256:
                failures.append(
                    f"MISMATCH: {art.path} (expected {art.sha256[:12]}... got {actual[:12]}...)"
                )
        return failures

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> PipelineManifest:
        data = json.loads(text)
        artifacts = [ArtifactEntry(**a) for a in data.pop("artifacts", [])]
        return cls(**data, artifacts=artifacts)

    @classmethod
    def load(cls, path: Path) -> PipelineManifest:
        return cls.from_json(path.read_text())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
