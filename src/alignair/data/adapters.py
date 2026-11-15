"""Source adapters into the shared training sample/batch contract."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from ..errors import AlignAIRError, ConfigurationError, TrainingDataError, error_context
from ..network.orientation import NUM_ORIENTATIONS, apply_orientation
from ..core.anchors import ANCHOR_GENES, ANCHOR_KEYS, CODON
from ..core.d_presence import ABSENT, D_PRESENCE
from ..core.boundaries import read_window
from ..reference.frames import gene_frames
from .batching import PositionLabels, TrainingBatch, TrainingSample, collate_samples
from .chip_inputs import chip_batch_inputs, chip_record_targets
from .novel_bank import batch_bank
from .targets import UNSCORABLE, build_targets, read_anchors, read_d_presence

if TYPE_CHECKING:
    from ..core.config import AlignAIRConfig


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


def _alleles(reference, gene, calls) -> torch.Tensor:
    """The multi-hot target of a record's ``calls`` over the reference's alleles of ``gene``; TrainingDataError for a
    call the reference does not hold."""
    vocabulary = reference.gene(gene.upper())
    encoded = torch.zeros(len(vocabulary.names), dtype=torch.float32)
    with error_context(gene=gene.upper()):
        for name in calls:
            position = vocabulary.index.get(name.strip())
            if position is None:
                raise TrainingDataError("training allele is not in the model reference",
                                        context={"allele": name.strip()})
            encoded[position] = 1
    return encoded


def _window(sample, width, genes, null_slot=None):
    """Retain a prefix and intersect model targets with it; source records remain untouched. An anchor target whose
    codon the prefix cuts becomes ``null_slot`` (not observed)."""
    if len(sample.tokens) <= width:
        return sample
    targets = dict(sample.targets)
    for key, value in targets.items():
        if isinstance(value, PositionLabels):
            targets[key] = PositionLabels(value.values[:width], value.padding)
    for key in ANCHOR_KEYS.values():
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
    width = read_window(config.max_seq_length)
    if overflow not in ("prefix", "error"):
        raise ConfigurationError("overflow must be 'prefix' or 'error'")
    if len(sample.tokens) > width and overflow == "error":
        raise TrainingDataError("training record exceeds the model window",
                                context={"length": len(sample.tokens), "max_length": width},
                                hint="Crop with matching annotations, or select the explicit prefix-window policy.")
    return _window(sample, width, genes, null_slot=config.max_seq_length)


def _anchor_targets(anchors, config) -> dict:
    """``v_anchor``/``j_anchor``: each anchor's read position (forward frame, like the boundaries), or the null slot
    ``max_seq_length`` when it is not observed; and ``{key}_supervise``, False for an unscorable anchor (its segment's
    alignment cannot be mapped), which the loss leaves out (its target is the null slot, never read)."""
    null, targets = config.max_seq_length, {}
    for gene in ANCHOR_GENES:
        key, position = ANCHOR_KEYS[gene], anchors[gene]
        targets[key] = torch.tensor(null if position in (None, UNSCORABLE) else position, dtype=torch.long)
        targets[f"{key}_supervise"] = torch.tensor(position != UNSCORABLE)
    return targets


def _d_presence_targets(record, config) -> dict:
    """``d_presence``: the class of the D the model's window shows (the crop cuts the read to it); and
    ``d_presence_supervise``, False for an unscorable D (its alignment cannot be mapped), which the loss leaves out
    (its target is absent, never read)."""
    klass = read_d_presence(record, read_window(config.max_seq_length))
    return {D_PRESENCE: torch.tensor(ABSENT if klass == UNSCORABLE else klass, dtype=torch.long),
            f"{D_PRESENCE}_supervise": torch.tensor(klass != UNSCORABLE)}


@dataclass(frozen=True)
class NucleotideBatchAdapter:
    reference: object
    config: AlignAIRConfig
    device: str = "cpu"
    augment_orientation: bool | None = None
    overflow: str = "prefix"

    def __call__(self, records) -> TrainingBatch:
        samples = []
        genes = _genes(self.config)
        records = list(records)
        bank = batch_bank(self.reference, records)       # None unless a read comes from a novel allele
        reference = self.reference if bank is None else bank.reference
        frames = gene_frames(reference)
        for i, record in enumerate(records):
            with _record_context(i, record):
                if bank is not None:
                    record = bank.resolve(record)
                bundle = build_targets(record, reference, self.config.has_d)
                if bank is not None:                     # a novel allele the read cannot tell from its family
                    for gene, (start, end) in bundle["germline"].items():
                        calls = bundle["calls"].get(gene.upper())
                        if calls:
                            calls.update(*(bank.ties(gene, call, (start, end)) for call in list(calls)))
                targets = {}
                for gene in genes:
                    present = gene in bundle["germline"]
                    supervise = present and not (gene == "d" and bundle["d_inverted"])
                    start, end = bundle["inseq"].get(gene, (0, 0))
                    targets[f"{gene}_start"] = torch.tensor([float(start)], dtype=torch.float32)
                    targets[f"{gene}_end"] = torch.tensor([float(end)], dtype=torch.float32)
                    targets[f"{gene}_supervise"] = torch.tensor(supervise)
                    targets[f"{gene}_allele"] = _alleles(
                        reference, gene, bundle["calls"].get(gene.upper(), ()) if supervise else ())
                for key in ("mutation_rate", "indel_count"):
                    targets[key] = torch.tensor([bundle[key]], dtype=torch.float32)
                targets["state_labels"] = PositionLabels(bundle["state_labels"])
                targets.update(_anchor_targets(read_anchors(record, reference), self.config))   # forward frame
                if self.config.has_d:
                    targets.update(_d_presence_targets(record, self.config))
                targets.update(chip_record_targets(record, reference, frames, bundle["region_labels"]))
                sample = TrainingSample(bundle["tokens"], targets, record.get("sequence_id"))
                samples.append(_limit(sample, self.config, genes, self.overflow))
        batch = collate_samples(samples, max_length=self.config.max_seq_length, reserve_terminal_slot=True)
        batch.inputs.update(chip_batch_inputs(batch.targets, reference,     # forward frame, like the targets
                                              paired=None if bank is None else bank.paired()))
        if bank is not None:
            batch.inputs.update({key: value.to(self.device) for key, value in bank.inputs(frames).items()})
        augment = True if self.augment_orientation is None else self.augment_orientation
        labels = (torch.randint(0, NUM_ORIENTATIONS, (len(samples),)) if augment
                  else torch.zeros(len(samples), dtype=torch.long))
        batch.inputs["tokenized_sequence"] = apply_orientation(
            batch.inputs["tokenized_sequence"], batch.token_mask, labels)
        batch.inputs["orientation"] = labels
        batch.targets["orientation"] = labels
        return batch.to(self.device)



def build_nucleotide_batch(records, reference_set, config, device="cpu", augment_orientation=None):
    return tuple(NucleotideBatchAdapter(reference_set, config, device, augment_orientation)(records))
