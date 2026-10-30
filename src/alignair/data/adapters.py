"""Source adapters into the shared training sample/batch contract."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from contextlib import contextmanager
from operator import index
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..errors import AlignAIRError, ConfigurationError, TrainingDataError, error_context
from ..neural.heads.orientation import NUM_ORIENTATIONS, apply_orientation
from ..neural.heads.region import READ_REGIONS, REGION_INDEX
from ..core.boundaries import read_window
from .batching import PositionLabels, TrainingBatch, TrainingSample, collate_samples
from .encoders import encode_allele_sets
from .projection import frame_offset_of, project_codons, projection_segments, translated_record_usable
from .targets import build_targets, read_anchors
from .tokenizer import AAProteinTokenizer

if TYPE_CHECKING:
    from ..core.config import AlignAIRConfig


# The anchor targets (the V Cys and J Trp/Phe codon starts) and the length of the codon each one starts.
ANCHOR_TARGETS = ("v_anchor", "j_anchor")
CODON = 3


def _genes(config):
    return ("v", "j", "d") if config.has_d else ("v", "j")


@contextmanager
def _record_context(i, record):
    if not isinstance(record, Mapping):
        raise TrainingDataError("training records must be mappings", context={"record_index": i})
    with error_context(stage="batch_adaptation", record_index=i, record_id=record.get("sequence_id")):
        try:
            yield
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AlignAIRError):
                raise
            raise TrainingDataError("record does not satisfy the batch adapter's schema",
                                    context={"field": exc.args[0] if isinstance(exc, KeyError) else None}) from exc


def _alleles(reference, gene, calls):
    vocabulary = reference.gene(gene.upper())
    with error_context(gene=gene.upper()):
        return torch.from_numpy(encode_allele_sets(
            [[name.strip() for name in calls]], vocabulary.index, len(vocabulary.names), strict=True)[0])


def _productive_supervise(known, config):
    """A known productivity label is supervised; an unknown one (labelled 0) only under the historical objective,
    which trains it as non-productive (``AlignAIRConfig.productive_unknown``)."""
    return torch.tensor(bool(known) or getattr(config, "productive_unknown", "negative") == "negative")


def _window(sample, width, genes, null_slot=None):
    """Retain a prefix and intersect model targets with it; source records remain untouched. An anchor target whose
    codon the prefix cuts becomes ``null_slot`` (not observed)."""
    if len(sample.tokens) <= width:
        return sample
    targets = dict(sample.targets)
    for key, value in targets.items():
        if isinstance(value, PositionLabels):
            targets[key] = PositionLabels(value.values[:width], value.padding)
    for key in ANCHOR_TARGETS:
        if key in targets and targets[key].item() + CODON > width:
            targets[key] = torch.tensor(null_slot, dtype=torch.long)
    for gene in genes:
        start, end = float(targets[f"{gene}_start"].item()), float(targets[f"{gene}_end"].item())
        visible = start < width and end > 0 and end > start
        targets[f"{gene}_start"] = torch.tensor([min(width, max(0., start)) if visible else 0.], dtype=torch.float32)
        targets[f"{gene}_end"] = torch.tensor([min(width, end) if visible else 0.], dtype=torch.float32)
        if not visible:
            targets[f"{gene}_supervise"] = torch.tensor(False)
            targets[f"{gene}_allele"] = torch.zeros_like(targets[f"{gene}_allele"])
    return TrainingSample(sample.tokens[:width], targets, sample.record_id)


def _limit(sample, config, genes, overflow):
    width = read_window(config.max_seq_length, getattr(config, "boundary_mode", "residue"))
    if overflow not in ("prefix", "error"):
        raise ConfigurationError("overflow must be 'prefix' or 'error'")
    if len(sample.tokens) > width and overflow == "error":
        raise TrainingDataError("training record exceeds the model window",
                                context={"length": len(sample.tokens), "max_length": width},
                                hint="Crop with matching annotations, or select the explicit prefix-window policy.")
    return _window(sample, width, genes, null_slot=config.max_seq_length)


def _anchor_targets(anchors, config) -> dict:
    """``v_anchor``/``j_anchor``: each anchor's read position (forward frame, like the boundaries), or the null slot
    ``max_seq_length`` when it is not observed."""
    null = config.max_seq_length
    return {f"{gene}_anchor": torch.tensor(null if anchors[gene] is None else anchors[gene], dtype=torch.long)
            for gene in ("v", "j")}


@dataclass(frozen=True)
class NucleotideBatchAdapter:
    reference: object
    config: AlignAIRConfig
    device: str = "cpu"
    augment_orientation: bool | None = None
    overflow: str = "prefix"

    def __call__(self, records) -> TrainingBatch:
        if getattr(self.config, "classification_mode", "multilabel") != "multilabel":
            raise ConfigurationError("nucleotide record adaptation requires multilabel targets")
        samples = []
        genes = _genes(self.config)
        for i, record in enumerate(records):
            with _record_context(i, record):
                bundle = build_targets(record, self.reference, self.config.has_d)
                targets = {}
                for gene in genes:
                    present = gene in bundle["germline"]
                    supervise = present and not (gene == "d" and bundle["d_inverted"])
                    start, end = bundle["inseq"].get(gene, (0, 0))
                    targets[f"{gene}_start"] = torch.tensor([float(start)], dtype=torch.float32)
                    targets[f"{gene}_end"] = torch.tensor([float(end)], dtype=torch.float32)
                    targets[f"{gene}_supervise"] = torch.tensor(supervise)
                    targets[f"{gene}_allele"] = _alleles(
                        self.reference, gene, bundle["calls"].get(gene.upper(), ()) if supervise else ())
                for key in ("mutation_rate", "indel_count", "productive"):
                    targets[key] = torch.tensor([bundle[key]], dtype=torch.float32)
                targets["productive_supervise"] = _productive_supervise(bundle["productive_known"], self.config)
                if getattr(self.config, "num_chain_types", 1) > 1:
                    targets["chain_type"] = torch.tensor(bundle["chain_type"], dtype=torch.long)
                targets["state_labels"] = PositionLabels(bundle["state_labels"])
                if getattr(self.config, "region_head", False):
                    # forward frame, like the boundaries; the head's classes are READ_REGIONS (no pad label)
                    targets["region_labels"] = PositionLabels(bundle["region_labels"] - REGION_INDEX[READ_REGIONS[0]])
                if getattr(self.config, "anchor_heads", False):
                    targets.update(_anchor_targets(read_anchors(record, self.reference), self.config))   # forward frame
                sample = TrainingSample(bundle["tokens"], targets, record.get("sequence_id"))
                samples.append(_limit(sample, self.config, genes, self.overflow))
        batch = collate_samples(samples, max_length=self.config.max_seq_length,
                                reserve_terminal_slot=getattr(self.config, "boundary_mode", "residue") == "half_open",
                                reject_empty_sequences=self.config.reject_empty_sequences)
        if getattr(self.config, "use_orientation", True):
            augment = True if self.augment_orientation is None else self.augment_orientation
            labels = (torch.randint(0, NUM_ORIENTATIONS, (len(samples),)) if augment
                      else torch.zeros(len(samples), dtype=torch.long))
            batch.inputs["tokenized_sequence"] = apply_orientation(
                batch.inputs["tokenized_sequence"], batch.token_mask, labels)
            batch.inputs["orientation"] = labels
            batch.targets["orientation"] = labels
        elif self.augment_orientation:
            raise ConfigurationError("orientation augmentation requires the model's orientation head")
        return batch.to(self.device)



@dataclass(frozen=True)
class ProteinAlleleBatchAdapter:
    """Translated reads with projected boundaries and full nucleotide allele vocabularies."""
    reference: object
    config: AlignAIRConfig
    device: str = "cpu"
    tokenizer: object = None
    overflow: str = "prefix"

    def __call__(self, records) -> TrainingBatch:
        if getattr(self.config, "classification_mode", "multilabel") != "multilabel":
            raise ConfigurationError("protein allele adaptation requires multilabel targets")
        if getattr(self.config, "use_orientation", False):
            raise ConfigurationError("protein records cannot use nucleotide orientation correction")
        tokenizer = self.tokenizer or AAProteinTokenizer()
        genes, samples = _genes(self.config), []
        for i, record in enumerate(records):
            with _record_context(i, record):
                if record.get("junction_start") is None or not record.get("sequence_aa"):
                    raise TrainingDataError("protein records require a translation and junction-derived frame")
                projection = project_codons(len(record["sequence"]), frame_offset_of(record["junction_start"]),
                                            projection_segments(record))
                encoded = tokenizer.encode(record["sequence_aa"])
                if len(encoded) != projection.n_residues:
                    raise TrainingDataError("translation length does not match the projected residue frame",
                                            context={"field": "sequence_aa"})
                targets = {}
                for gene in genes:
                    bounds = projection.aa_bounds.get(gene.upper())
                    supervise = bounds is not None and not (gene == "d" and record.get("d_inverted", False))
                    start, end = bounds if bounds is not None else (0, 0)
                    targets[f"{gene}_start"] = torch.tensor([float(start)], dtype=torch.float32)
                    targets[f"{gene}_end"] = torch.tensor([float(end)], dtype=torch.float32)
                    targets[f"{gene}_supervise"] = torch.tensor(supervise)
                    calls = str(record.get(f"{gene}_call") or "").split(",") if supervise else ()
                    targets[f"{gene}_allele"] = _alleles(self.reference, gene, calls)
                for key, value in (("mutation_rate", record["mutation_rate"]),
                                   ("indel_count", record.get("n_indels", 0)),
                                   ("productive", float(bool(record.get("productive"))))):
                    targets[key] = torch.tensor([value], dtype=torch.float32)
                targets["productive_supervise"] = _productive_supervise(record.get("productive") is not None,
                                                                        self.config)
                sample = TrainingSample(encoded, targets, record.get("sequence_id"))
                samples.append(_limit(sample, self.config, genes, self.overflow))
        return collate_samples(samples, max_length=self.config.max_seq_length, device=self.device,
                               reserve_terminal_slot=getattr(self.config, "boundary_mode", "residue") == "half_open")


@dataclass(frozen=True)
class CategoricalBatchAdapter:
    """Prepared class-index/compatible-set samples, independent of their sequence alphabet."""
    max_length: int
    device: str = "cpu"
    class_counts: dict[str, int] | None = None
    reserve_terminal_slot: bool = True

    @classmethod
    def from_config(cls, config, *, device="cpu"):
        return cls(config.max_seq_length, device,
                   {spec.name: spec.allele_count for spec in config.gene_specs},
                   config.boundary_mode == "half_open")

    def __call__(self, samples) -> TrainingBatch:
        samples = list(samples)
        if not samples:
            raise TrainingDataError("cannot adapt an empty categorical batch")
        for i, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise TrainingDataError("categorical samples must be mappings", context={"record_index": i})
        genes = tuple(gene for gene in ("v", "j", "d") if f"{gene}_class" in samples[0])
        if not genes:
            raise TrainingDataError("categorical samples need gene class labels")
        validity = {gene for gene in genes if gene == "d" or any(f"{gene}_valid" in row for row in samples)}
        supervision_masks = {f"{gene}_{suffix}" for gene in genes for suffix in ("supervise", "class_supervise")
                             if any(f"{gene}_{suffix}" in row for row in samples)}
        compatible = {gene for gene in genes if any(f"{gene}_compatible" in row for row in samples)}
        prepared = []
        for i, sample in enumerate(samples):
            with _record_context(i, sample):
                if tuple(gene for gene in ("v", "j", "d") if f"{gene}_class" in sample) != genes:
                    raise TrainingDataError("categorical samples have different gene sets")
                targets = {}
                for gene in genes:
                    value = sample[f"{gene}_class"]
                    if isinstance(value, bool):
                        raise TrainingDataError("class indices must be integers", context={"field": f"{gene}_class"})
                    targets[f"{gene}_class"] = torch.tensor(index(value), dtype=torch.long)
                    for boundary in ("start", "end"):
                        targets[f"{gene}_{boundary}"] = torch.tensor([float(sample[f"{gene}_{boundary}"])], dtype=torch.float32)
                    if gene in validity:
                        targets[f"{gene}_valid"] = torch.tensor(float(sample.get(f"{gene}_valid", 1.)), dtype=torch.float32)
                    for key in (f"{gene}_supervise", f"{gene}_class_supervise"):
                        if key in supervision_masks:
                            mask = sample.get(key, True)
                            if not isinstance(mask, (bool, int, float, np.number)) or mask not in (0, 1):
                                raise TrainingDataError("supervision must be binary", context={"field": key})
                            targets[key] = torch.tensor(bool(mask))
                    key = f"{gene}_compatible"
                    if gene in compatible:
                        if self.class_counts is None or gene not in self.class_counts:
                            raise ConfigurationError("compatible sets require explicit per-gene class counts")
                        if key not in sample:
                            raise TrainingDataError("compatible-set annotation is missing", context={"field": key})
                        membership = torch.zeros(self.class_counts[gene], dtype=torch.float32)
                        for value in sample[key]:
                            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < membership.numel():
                                raise TrainingDataError("compatible class is outside the vocabulary", context={"field": key})
                            membership[value] = 1
                        targets[key] = membership
                prepared.append(TrainingSample(sample["tokens"], targets, sample.get("sequence_id")))
        return collate_samples(prepared, max_length=self.max_length, device=self.device,
                               reserve_terminal_slot=self.reserve_terminal_slot)


def build_nucleotide_batch(records, reference_set, config, device="cpu", augment_orientation=None):
    return tuple(NucleotideBatchAdapter(reference_set, config, device, augment_orientation)(records))


def build_protein_allele_batch(records, reference_set, config, device="cpu", tokenizer=None):
    return tuple(ProteinAlleleBatchAdapter(reference_set, config, device, tokenizer)(records))


def collate_categorical(samples, max_len, device="cpu"):
    return tuple(CategoricalBatchAdapter(max_len, device)(samples))
