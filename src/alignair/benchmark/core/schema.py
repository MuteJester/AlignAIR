"""Benchmark data structures for simulated AIRR alignment evaluation.

The benchmark stores both:
  - canonical truth: the forward GenAIRR frame after any crop;
  - presented truth: coordinates/labels in the sequence actually given to a tool
    after optional orientation transforms.

This lets the same dataset evaluate neural models that canonicalize internally and
external tools that report coordinates on the presented query.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GENES = ("v", "d", "j")
ORIENTATION_NAMES = {
    0: "identity",
    1: "reverse_complement",
    2: "complement",
    3: "reverse",
}


@dataclass(frozen=True)
class GeneTruth:
    """Ground truth for one V/D/J segment."""

    calls: tuple[str, ...] = ()
    primary: str | None = None
    sequence_start: int | None = None
    sequence_end: int | None = None
    germline_start: int | None = None
    germline_end: int | None = None

    @property
    def present(self) -> bool:
        return bool(self.calls) and self.sequence_start is not None and self.sequence_end is not None

    @property
    def gene_names(self) -> tuple[str, ...]:
        return tuple(c.split("*")[0] for c in self.calls)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneTruth":
        return cls(
            calls=tuple(data.get("calls", ())),
            primary=data.get("primary"),
            sequence_start=data.get("sequence_start"),
            sequence_end=data.get("sequence_end"),
            germline_start=data.get("germline_start"),
            germline_end=data.get("germline_end"),
        )


@dataclass
class BenchmarkCase:
    """One benchmark input sequence plus complete GenAIRR-derived truth."""

    case_id: str
    stratum: str
    sequence: str
    canonical_sequence: str
    orientation_id: int
    genes: dict[str, GeneTruth]
    presented_genes: dict[str, GeneTruth]
    region_labels: list[int] = field(default_factory=list)
    state_labels: list[int] = field(default_factory=list)
    presented_region_labels: list[int] = field(default_factory=list)
    presented_state_labels: list[int] = field(default_factory=list)
    scalars: dict[str, float] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)

    def truth(self, frame: str = "canonical") -> dict[str, GeneTruth]:
        if frame == "canonical":
            return self.genes
        if frame == "presented":
            return self.presented_genes
        raise ValueError("frame must be 'canonical' or 'presented'")

    def labels(self, name: str, frame: str = "canonical") -> list[int]:
        if name == "region":
            return self.region_labels if frame == "canonical" else self.presented_region_labels
        if name == "state":
            return self.state_labels if frame == "canonical" else self.presented_state_labels
        raise ValueError("name must be 'region' or 'state'")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["genes"] = {g: asdict(v) for g, v in self.genes.items()}
        data["presented_genes"] = {g: asdict(v) for g, v in self.presented_genes.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkCase":
        d = dict(data)
        d["genes"] = {g: GeneTruth.from_dict(v) for g, v in d.get("genes", {}).items()}
        d["presented_genes"] = {
            g: GeneTruth.from_dict(v) for g, v in d.get("presented_genes", {}).items()
        }
        return cls(**d)


@dataclass(frozen=True)
class StratumSpec:
    """A named GenAIRR corruption/cropping/orientation slice."""

    name: str
    n: int
    progress: float
    crop_to: int | None = None
    orientation_ids: tuple[int, ...] = (0,)
    seed_offset: int = 0
    param_overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkSpec:
    """Generation recipe for a benchmark dataset."""

    name: str
    dataconfig_name: str
    seed: int
    strata: tuple[StratumSpec, ...]
    version: str = "0.1"
    description: str = ""

    @property
    def n_cases(self) -> int:
        return sum(s.n for s in self.strata)
