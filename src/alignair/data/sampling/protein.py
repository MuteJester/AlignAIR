"""One protein-class sample builder with explicit scientific acceptance policies."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from operator import index

from ...errors import ConfigurationError, TrainingDataError, error_context
from ..projection import frame_offset_of, project_codons, projection_segments, protein_translation_issue
from .contracts import SampleResult

REJECT_REASONS = ("not_productive", "no_aa", "no_frame", "stop_codon", "unknown_allele",
                  "no_bounds", "too_long", "empty", "translation_length_mismatch")


@dataclass(frozen=True)
class ProteinSamplePolicy:
    require_productive: bool = True
    allow_stops: bool = False

    def __post_init__(self):
        if type(self.require_productive) is not bool or type(self.allow_stops) is not bool:
            raise ConfigurationError("protein acceptance flags must be booleans")


@dataclass(frozen=True)
class ProteinSampleBuilder:
    class_maps: Mapping
    tokenizer: object
    max_length: int
    d_index: Mapping | None = None
    policy: ProteinSamplePolicy = field(default_factory=ProteinSamplePolicy)

    def __post_init__(self):
        try:
            length = index(self.max_length)
            if isinstance(self.max_length, bool) or length <= 1:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("sample window must be a positive integer with room for the read and endpoint") from exc
        if not isinstance(self.policy, ProteinSamplePolicy):
            raise ConfigurationError("supply an explicit ProteinSamplePolicy")
        if not isinstance(self.class_maps, Mapping) or any(
                not callable(getattr(self.class_maps.get(gene), "class_of", None)) for gene in ("V", "J")):
            raise ConfigurationError("protein samples require V and J class maps")
        if not callable(getattr(self.tokenizer, "encode", None)):
            raise ConfigurationError("sample tokenizer must provide encode")
        if self.d_index is not None:
            if (not isinstance(self.d_index, Mapping)
                    or any(not isinstance(name, str) or not name for name in self.d_index)
                    or any(isinstance(value, bool) or not isinstance(value, Integral) for value in self.d_index.values())
                    or set(self.d_index.values()) != set(range(len(self.d_index)))):
                raise ConfigurationError("D allele indices must form a contiguous vocabulary starting at zero")
        object.__setattr__(self, "max_length", length)

    def __call__(self, record) -> SampleResult:
        if not isinstance(record, Mapping):
            raise TrainingDataError("sample records must be mappings")
        with error_context(stage="sample_preparation", record_id=record.get("sequence_id")):
            if self.policy.require_productive and not record.get("productive"):
                return SampleResult(None, "not_productive")
            aa = record.get("sequence_aa")
            if aa is not None and not isinstance(aa, str):
                raise TrainingDataError("protein sequence must be a string", context={"field": "sequence_aa"})
            if not aa:
                return SampleResult(None, "no_aa")
            if record.get("junction_start") is None:
                return SampleResult(None, "no_frame")
            if not self.policy.allow_stops and "*" in aa:
                return SampleResult(None, "stop_codon")
            issue = protein_translation_issue(record)
            if issue is not None:
                return SampleResult(None, issue)
            try:
                labels = {gene: self.class_maps[gene.upper()].class_of(
                    str(record.get(f"{gene}_call") or "").split(",")[0].strip()) for gene in ("v", "j")}
            except KeyError:
                return SampleResult(None, "unknown_allele")
            projection = project_codons(len(record["sequence"]), frame_offset_of(record["junction_start"]),
                                        projection_segments(record))
            if any(projection.aa_bounds.get(gene) is None for gene in ("V", "J")):
                return SampleResult(None, "no_bounds")
            tokens = self.tokenizer.encode(aa)
            if not len(tokens):
                return SampleResult(None, "empty")
            if len(tokens) != len(aa):
                raise TrainingDataError("protein tokenization must preserve residue positions",
                                        context={"field": "tokens"})
            if len(tokens) >= self.max_length:
                return SampleResult(None, "too_long")
            sample = {"tokens": tokens}
            for gene in ("v", "j"):
                sample[f"{gene}_class"] = labels[gene]
                sample[f"{gene}_start"], sample[f"{gene}_end"] = projection.aa_bounds[gene.upper()]
            if self.d_index:
                bounds = projection.aa_bounds.get("D")
                call = str(record.get("d_call") or "").split(",")[0].strip()
                valid = not record.get("d_inverted", False) and call in self.d_index and bounds is not None
                sample["d_class"] = self.d_index[call] if valid else -1
                sample["d_start"], sample["d_end"] = bounds if valid else (0, 0)
                sample["d_valid"] = float(valid)
            return SampleResult(sample)


def prepare_protein_sample(rec, class_maps, tokenizer, max_len, *, d_index=None, policy=None):
    """Functional entry point; repeated preparation should reuse a ProteinSampleBuilder."""
    return ProteinSampleBuilder(class_maps, tokenizer, max_len, d_index,
                                ProteinSamplePolicy() if policy is None else policy)(rec)
