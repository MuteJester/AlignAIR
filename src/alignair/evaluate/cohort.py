"""Ordered, supervision-aware alignment observations over caller-owned cohort records."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import islice
import math
from typing import NamedTuple

import torch

from ..core.config import AlignAIRConfig
from ..data.batching import BatchAdapter
from ..errors import ConfigurationError, NumericalError, TrainingDataError, error_context
from ..tasks.supervision import categorical_rows, supervised_rows
from ..core.runtime import evaluation_mode
from .statistics import _integer


class ClassificationObservation(NamedTuple):
    record_index: int
    truth: tuple[int, ...]
    prediction: int

    @property
    def correct(self) -> bool:
        """Membership correctness for exact, compatible, or multilabel truth."""
        return self.prediction in self.truth


class BoundaryObservation(NamedTuple):
    record_index: int
    truth: float
    prediction: float
    absolute_error: float


class PredictionSetObservation(NamedTuple):
    record_index: int
    members: tuple[int, ...]


class ScalarObservation(NamedTuple):
    record_index: int
    truth: float
    prediction: float
    absolute_error: float


@dataclass(frozen=True)
class CohortObservations:
    """Detached Python observations; indices refer to the original ordered record stream."""
    record_count: int
    classification: dict[str, tuple[ClassificationObservation, ...]]
    boundaries: dict[str, tuple[BoundaryObservation, ...]]
    prediction_sets: dict[str, tuple[PredictionSetObservation, ...]] = field(default_factory=dict)
    scalars: dict[str, tuple[ScalarObservation, ...]] = field(default_factory=dict)

    @property
    def support(self) -> dict:
        support = {"classification": {gene: len(rows) for gene, rows in self.classification.items()},
                   "boundaries": {key: len(rows) for key, rows in self.boundaries.items()}}
        if self.scalars:
            support["scalars"] = {key: len(rows) for key, rows in self.scalars.items()}
        return support


def _tensor(mapping, key, shape, device=None):
    value = mapping.get(key)
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) not in shape
            or device is not None and value.device != device or value.is_complex()):
        raise TrainingDataError("cohort field has an invalid tensor shape, type, or device",
                                context={"field": key, "expected_shapes": shape})
    return value


def _finite(values, selected, field):
    if not torch.isfinite(values[selected]).all():
        raise NumericalError("non-finite supervised cohort values", context={"field": field})


def _threshold(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or not 0 <= value <= 1):
        raise ConfigurationError("prediction-set threshold must be between zero and one",
                                 context={"field": "prediction_set_threshold"})
    return value


def _member_sets(membership):
    members = [[] for _ in range(membership.shape[0])]
    for row, label in membership.nonzero().cpu().tolist():
        members[row].append(label)
    return [tuple(labels) for labels in members]


def allele_observations(probabilities, targets, *, supervision=None, prediction_set_threshold=None,
                        record_offset: int = 0):
    """Shared tensor-to-observation adapter for multilabel allele heads and historical counters."""
    threshold = _threshold(prediction_set_threshold)
    record_offset = _integer(record_offset, "record_offset", 0)
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2
            or probabilities.shape[1] == 0 or not probabilities.is_floating_point()):
        raise TrainingDataError("allele probabilities must be a floating-point batch-by-vocabulary matrix")
    targets = _tensor({"alleles": targets}, "alleles", (tuple(probabilities.shape),), probabilities.device)
    masks = {} if supervision is None else {"allele_supervise": supervision}
    valid = supervised_rows(masks, "allele", probabilities)
    if not ((targets[valid] == 0) | (targets[valid] == 1)).all():
        raise TrainingDataError("multilabel cohort truth must contain binary membership", context={"field": "alleles"})
    selected = valid & targets.bool().any(-1)
    scores = probabilities[selected]
    if not ((scores >= 0) & (scores <= 1)).all():
        raise NumericalError("multilabel cohort scores must be probabilities", context={"field": "probabilities"})
    _finite(probabilities, selected, "probabilities")
    indices = selected.nonzero().flatten().cpu().tolist()
    truth = _member_sets(targets[selected])
    predictions = scores.argmax(-1).cpu().tolist()
    rows = [ClassificationObservation(record_offset + row, labels, prediction)
            for row, labels, prediction in zip(indices, truth, predictions)]
    sets = ([] if threshold is None else
            [PredictionSetObservation(record_offset + row, members)
             for row, members in zip(indices, _member_sets(scores > threshold))])
    return rows, sets


def _classification(outputs, targets, config, gene, count, class_count, offset, threshold):
    key = config.classification_key(gene)
    scores = _tensor(outputs, key, ((count, class_count),))
    if not scores.is_floating_point():
        raise TrainingDataError("classification scores must be floating-point tensors", context={"field": key})
    valid = supervised_rows(targets, gene, scores)
    categorical = config.classification_mode == "categorical"
    membership_key = f"{gene}_compatible" if categorical else f"{gene}_allele"
    membership = None
    if not categorical or membership_key in targets:
        membership = _tensor(targets, membership_key, ((count, class_count),), scores.device)
    else:
        _tensor(targets, f"{gene}_class", ((count,),), scores.device)
    if not categorical:
        rows, sets = allele_observations(scores, membership, supervision=valid,
                                         prediction_set_threshold=threshold, record_offset=offset)
        return rows, valid, scores.device, sets
    selected = categorical_rows(targets, gene, scores)
    _finite(scores, selected, key)
    indices = selected.nonzero().flatten().cpu().tolist()
    predictions = scores[selected].argmax(-1).cpu().tolist()
    if membership is None:
        truth = [(label,) for label in targets[f"{gene}_class"][selected].cpu().tolist()]
    else:
        truth = _member_sets(membership[selected])
    rows = [ClassificationObservation(offset + row, labels, prediction)
            for row, labels, prediction in zip(indices, truth, predictions)]
    return rows, valid, scores.device, []


def _scalar_observations(outputs, targets, key, count, offset, *, device=None, selected=None,
                         observation_type=ScalarObservation):
    shape = ((count,), (count, 1))
    predicted = _tensor(outputs, key, shape, device).reshape(-1)
    truth = _tensor(targets, key, shape, predicted.device).reshape(-1)
    if predicted.dtype == torch.bool or truth.dtype == torch.bool:
        raise TrainingDataError("scalar observations must be numeric, not boolean", context={"field": key})
    selected = supervised_rows(targets, key, predicted) if selected is None else selected
    _finite(predicted, selected, key)
    _finite(truth, selected, key)
    errors = (predicted[selected] - truth[selected]).abs()
    _finite(errors, slice(None), key)
    return [observation_type(offset + row, target, prediction, error)
            for row, target, prediction, error in zip(selected.nonzero().flatten().cpu().tolist(),
                truth[selected].cpu().tolist(), predicted[selected].cpu().tolist(), errors.cpu().tolist())]


def collect_alignment_observations(model, records, batch_adapter: BatchAdapter, config: AlignAIRConfig,
                                   *, batch_size: int = 128, genes=None, prediction_set_threshold=None,
                                   scalar_heads=()) -> CohortObservations:
    """Evaluate an ordered record stream through a shared batch adapter, including its final tail.

    The adapter must preserve record order and cardinality and place inputs/targets on the model's
    device. Gene masks govern boundaries; categorical class masks/unknown labels and empty allele
    sets separately govern classification. Truth sets remain sets; no canonical label is guessed.
    Only native Python observations are retained, never model outputs or gradient graphs. Source
    freezing, task-specific reports, and strata belong to the caller. Thresholded allele sets and
    named scalar heads are opt-in, so default collection retains no extra output data.
    """
    batch_size = _integer(batch_size, "batch_size", 1)
    if not isinstance(model, torch.nn.Module) or not isinstance(config, AlignAIRConfig):
        raise ConfigurationError("cohort collection requires a torch model and AlignAIRConfig")
    if not callable(batch_adapter):
        raise ConfigurationError("batch_adapter must be callable", context={"field": "batch_adapter"})
    threshold = _threshold(prediction_set_threshold)
    if threshold is not None and config.classification_mode != "multilabel":
        raise ConfigurationError("thresholded allele sets require multilabel model heads")
    try:
        heads = tuple(scalar_heads)
        if isinstance(scalar_heads, str) or any(not isinstance(head, str) or not head for head in heads) or len(set(heads)) != len(heads):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("scalar_heads must contain distinct nonempty names") from exc
    canonical = {spec.name: spec.allele_count for spec in config.gene_specs}
    try:
        selected = tuple(canonical) if genes is None else tuple(genes)
        if isinstance(genes, str) or not selected or len(set(selected)) != len(selected) or not set(selected) <= canonical.keys():
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("genes must be a nonempty unique subset of configured gene names") from exc
    counts = {gene: _integer(canonical[gene], f"{gene}_class_count", 1) for gene in selected}
    classification = {gene: [] for gene in selected}
    boundaries = {f"{gene}_{boundary}": [] for gene in selected for boundary in ("start", "end")}
    prediction_sets = {} if threshold is None else {gene: [] for gene in selected}
    scalars = {head: [] for head in heads}
    total = 0
    with evaluation_mode(model), error_context(stage="cohort_collection"):
        stream = iter(records)
        while chunk := tuple(islice(stream, batch_size)):
            with error_context(record_offset=total, batch_size=len(chunk)):
                batch = batch_adapter(chunk)
                try:
                    inputs, targets = batch
                except (TypeError, ValueError) as exc:
                    raise TrainingDataError("cohort adapters must return an (inputs, targets) batch") from exc
                if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping):
                    raise TrainingDataError("cohort inputs and targets must be mappings")
                outputs = model(inputs)
                if not isinstance(outputs, Mapping):
                    raise TrainingDataError("cohort model outputs must be a mapping")
                for gene in selected:
                    with error_context(gene=gene):
                        rows, valid, device, sets = _classification(outputs, targets, config, gene, len(chunk), counts[gene], total, threshold)
                        classification[gene].extend(rows)
                        if threshold is not None:
                            prediction_sets[gene].extend(sets)
                        for boundary in ("start", "end"):
                            key = f"{gene}_{boundary}"
                            boundaries[key].extend(_scalar_observations(outputs, targets, key, len(chunk), total,
                                device=device, selected=valid, observation_type=BoundaryObservation))
                for head in heads:
                    with error_context(head=head):
                        scalars[head].extend(_scalar_observations(outputs, targets, head, len(chunk), total))
                total += len(chunk)
    return CohortObservations(total, {gene: tuple(rows) for gene, rows in classification.items()},
                              {key: tuple(rows) for key, rows in boundaries.items()},
                              {gene: tuple(rows) for gene, rows in prediction_sets.items()},
                              {head: tuple(rows) for head, rows in scalars.items()})
