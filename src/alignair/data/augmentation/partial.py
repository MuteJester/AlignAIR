"""Alignment-aware reference evidence and conservative cropped protein-class supervision."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING
import math

from ...errors import ConfigurationError, TrainingDataError, error_context
from ..alignment import query_reference_map
from ..coordinates import _coordinate, query_spans
from ..sampling.contracts import SampleResult
from ..projection import frame_offset_of
from .cropping import crop_window, shift_clip, fragment_compatible_classes
from .protein import aa_substitute, il_ambiguity

if TYPE_CHECKING:
    from numpy.random import Generator
    from ...reference.protein import ProteinReference
    from ..sampling import ProteinSampleBuilder

SAMPLE_POLICY_ID = "protein-crop-reference-fragments-v1"


@dataclass(frozen=True)
class ReferenceEvidence:
    fragments: tuple[str, ...]
    positions: tuple[int, ...]
    reason: str | None = None


def reference_residue_evidence(record, gene, reference, lo, hi) -> ReferenceEvidence:
    """Use only query codons aligned to three consecutive bases in the reference coding frame."""
    try:
        lo, hi = _coordinate(lo), _coordinate(hi)
        if not 0 <= lo < hi <= len(record["sequence_aa"]):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingDataError("reference evidence window is outside the protein read") from exc
    gene = gene.upper()
    with error_context(stage="crop_reference_projection", gene=gene, record_id=record.get("sequence_id")):
        spans = query_spans(record)
        if gene.lower() not in spans:
            return ReferenceEvidence((), (), "gene_absent")
        name = str(record.get(f"{gene.lower()}_call") or "").split(",")[0].strip()
        germline = reference.germlines.genes[gene].get(name)
        if germline is None:
            raise TrainingDataError("source allele is absent from the protein reference")
        start, end = (record.get(f"{gene.lower()}_germline_{side}") for side in ("start", "end"))
        if start is None or end is None:
            return ReferenceEvidence((), (), "missing_reference_coordinates")
        path = query_reference_map(record.get(f"{gene.lower()}_cigar"), spans[gene.lower()], (start, end))
        if path.reason is not None:
            return ReferenceEvidence((), (), path.reason)
        nucleotide_reference = reference.source.gene(gene)
        if _coordinate(end) > len(nucleotide_reference.sequences[nucleotide_reference.index[name]]):
            return ReferenceEvidence((), (), "reference_span_out_of_range")
        frame = frame_offset_of(record["junction_start"])
        fragments, current, positions = [], [], []
        previous_read = previous_reference = None
        for residue in range(lo, hi):
            query = frame + 3 * residue - path.query_start
            codon = path.positions[query:query + 3] if query >= 0 else ()
            mapped = (len(codon) == 3 and codon[0] is not None and codon == tuple(range(codon[0], codon[0] + 3))
                      and (codon[0] - germline.frame) % 3 == 0)
            position = (codon[0] - germline.frame) // 3 if mapped else -1
            if not mapped or not 0 <= position < len(germline.aa_seq) or germline.aa_seq[position] == "X":
                if current:
                    fragments.append("".join(current)); current = []
                previous_read = previous_reference = None
                continue
            if current and (residue != previous_read + 1 or position != previous_reference + 1):
                fragments.append("".join(current)); current = []
            current.append(germline.aa_seq[position])
            positions.append(position)
            previous_read, previous_reference = residue, position
        if current:
            fragments.append("".join(current))
        return ReferenceEvidence(tuple(fragments), tuple(positions), None if positions else "no_reference_codon_evidence")


def crop_protein_sample(record, base, reference, lo, hi):
    """Retain known boundaries; class-only masks distinguish unavailable reference evidence."""
    try:
        lo, hi = _coordinate(lo), _coordinate(hi)
        if not 0 <= lo < hi <= len(base["tokens"]):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingDataError("protein crop is outside the encoded read") from exc
    tokens = base["tokens"][lo:hi]
    result = {**base, "tokens": tokens.clone() if hasattr(tokens, "clone") else tokens.copy()}
    genes = tuple(gene for gene in ("v", "j", "d") if f"{gene}_class" in base)
    for gene in genes:
        original = (base[f"{gene}_start"], base[f"{gene}_end"])
        bounds = shift_clip(original, lo, hi)
        valid = bounds is not None and bool(base.get(f"{gene}_valid", 1.))
        result[f"{gene}_start"], result[f"{gene}_end"] = bounds if bounds is not None else (0, 0)
        result[f"{gene}_valid"] = float(valid)
        labels, reason, evidence_count = set(), "gene_absent", 0
        if valid and gene == "d":
            valid = lo <= original[0] and hi >= original[1]
            result["d_valid"] = float(valid)
            labels = {base["d_class"]} if valid else set()
            reason = "d_source_label" if valid else "partial_d"
        elif valid:
            evidence = reference_residue_evidence(record, gene, reference, lo, hi)
            reason, evidence_count = evidence.reason, len(evidence.positions)
            if evidence.fragments:
                prototypes = reference.prototype_sequences(gene)
                labels = fragment_compatible_classes(evidence.fragments, prototypes)
                if base[f"{gene}_class"] not in labels:
                    raise TrainingDataError("reference fragment targets exclude their source class", context={"gene": gene})
                reason = "reference_fragments"
        result[f"{gene}_compatible"] = labels
        result[f"{gene}_class_supervise"] = bool(labels)
        result[f"{gene}_class"] = base[f"{gene}_class"] if labels else -1
        result[f"{gene}_class_reason"] = reason
        result[f"{gene}_reference_evidence"] = evidence_count
    return result


@dataclass(frozen=True)
class ProteinCropPolicy:
    crop_probability: float = .4
    substitution_probability: float = .4
    il_probability: float = .15

    def __post_init__(self):
        for value in (self.crop_probability, self.substitution_probability, self.il_probability):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigurationError("augmentation probabilities must be finite numbers in [0, 1]")


@dataclass(frozen=True)
class ProteinCropAugmenter:
    reference: ProteinReference
    builder: ProteinSampleBuilder
    rng: Generator
    policy: ProteinCropPolicy = ProteinCropPolicy()

    def __post_init__(self):
        if not isinstance(self.policy, ProteinCropPolicy):
            raise ConfigurationError("supply a ProteinCropPolicy")
        for gene in ("V", "J"):
            if self.builder.class_maps[gene].to_json() != self.reference.class_maps[gene].to_json():
                raise ConfigurationError("crop builder and reference label spaces disagree", context={"gene": gene})
        if self.builder.d_index:
            if ("D" not in self.reference.source.genes
                    or dict(self.builder.d_index) != self.reference.source.gene("D").index):
                raise ConfigurationError("crop builder and reference D vocabularies disagree")

    def __call__(self, record):
        prepared = self.builder(record)
        if prepared.sample is None:
            return prepared
        base = prepared.sample
        if self.rng.random() < self.policy.crop_probability:
            n = len(base["tokens"])
            keep = int(self.rng.uniform(.4, .9) * n)
            lo, hi = crop_window(n, str(self.rng.choice(["nterm", "cterm", "internal"])), keep)
            result = crop_protein_sample(record, base, self.reference, lo, hi)
            sequence = record["sequence_aa"][lo:hi]
        else:
            result = dict(base)
            sequence = record["sequence_aa"]
            for gene in (g for g in ("v", "j", "d") if f"{g}_class" in base):
                valid = bool(base.get(f"{gene}_valid", 1.))
                result[f"{gene}_valid"] = float(valid)
                result[f"{gene}_class_supervise"] = valid
                result[f"{gene}_compatible"] = {base[f"{gene}_class"]} if valid else set()
                result[f"{gene}_class_reason"] = "source_label" if valid else "gene_absent"
                result[f"{gene}_reference_evidence"] = 0
        if self.rng.random() < self.policy.substitution_probability:
            sequence = aa_substitute(sequence, float(self.rng.uniform(.02, .12)), self.rng)
        if self.rng.random() < self.policy.il_probability:
            sequence = il_ambiguity(sequence)
        tokens = self.builder.tokenizer.encode(sequence)
        if len(tokens) != len(result["tokens"]):
            raise TrainingDataError("crop corruption must preserve residue positions")
        result["tokens"] = tokens
        return SampleResult(result)
