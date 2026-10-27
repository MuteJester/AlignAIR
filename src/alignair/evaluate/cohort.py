"""Ordered, supervision-aware alignment observations over caller-owned cohort records."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from typing import NamedTuple

import torch

from ..core.config import AlignAIRConfig
from ..data.batching import BatchAdapter
from ..errors import ConfigurationError, NumericalError, TrainingDataError, error_context
from ..tasks.supervision import categorical_rows, supervised_rows
from ..train.engine import evaluation_mode
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


@dataclass(frozen=True)
class CohortObservations:
    """Detached Python observations; indices refer to the original ordered record stream."""
    record_count: int
    classification: dict[str, tuple[ClassificationObservation, ...]]
    boundaries: dict[str, tuple[BoundaryObservation, ...]]

    @property
    def support(self) -> dict:
        return {"classification": {gene: len(rows) for gene, rows in self.classification.items()},
                "boundaries": {key: len(rows) for key, rows in self.boundaries.items()}}


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


def _classification(outputs, targets, config, gene, count, class_count, offset):
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
    if categorical:
        selected = categorical_rows(targets, gene, scores)
    else:
        if not ((membership[valid] == 0) | (membership[valid] == 1)).all():
            raise TrainingDataError("multilabel cohort truth must contain binary membership",
                                    context={"field": membership_key})
        selected = valid & membership.bool().any(-1)
        if not ((scores[selected] >= 0) & (scores[selected] <= 1)).all():
            raise NumericalError("multilabel cohort scores must be probabilities", context={"field": key})
    _finite(scores, selected, key)
    indices = selected.nonzero().flatten().cpu().tolist()
    predictions = scores[selected].argmax(-1).cpu().tolist()
    if membership is None:
        truth = [(label,) for label in targets[f"{gene}_class"][selected].cpu().tolist()]
    else:
        truth = [[] for _ in indices]
        for row, label in membership[selected].nonzero().cpu().tolist():
            truth[row].append(label)
        truth = [tuple(labels) for labels in truth]
    rows = [ClassificationObservation(offset + row, labels, prediction)
            for row, labels, prediction in zip(indices, truth, predictions)]
    return rows, valid, scores.device


def collect_alignment_observations(model, records, batch_adapter: BatchAdapter, config: AlignAIRConfig,
                                   *, batch_size: int = 128, genes=None) -> CohortObservations:
    """Evaluate an ordered record stream through a shared batch adapter, including its final tail.

    The adapter must preserve record order and cardinality and place inputs/targets on the model's
    device. Gene masks govern boundaries; categorical class masks/unknown labels and empty allele
    sets separately govern classification. Truth sets remain sets; no canonical label is guessed.
    Only native Python observations are retained, never model outputs or gradient graphs. Source
    freezing, task-specific reports, strata, and optional auxiliary heads belong to the caller.
    """
    batch_size = _integer(batch_size, "batch_size", 1)
    if not isinstance(model, torch.nn.Module) or not isinstance(config, AlignAIRConfig):
        raise ConfigurationError("cohort collection requires a torch model and AlignAIRConfig")
    if not callable(batch_adapter):
        raise ConfigurationError("batch_adapter must be callable", context={"field": "batch_adapter"})
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
                        rows, valid, device = _classification(outputs, targets, config, gene, len(chunk), counts[gene], total)
                        classification[gene].extend(rows)
                        indices = valid.nonzero().flatten().cpu().tolist()
                        for boundary in ("start", "end"):
                            key = f"{gene}_{boundary}"
                            shape = ((len(chunk),), (len(chunk), 1))
                            predicted = _tensor(outputs, key, shape, device).reshape(-1)
                            truth = _tensor(targets, key, shape, device).reshape(-1)
                            if predicted.dtype == torch.bool or truth.dtype == torch.bool:
                                raise TrainingDataError("boundary coordinates must be numeric, not boolean",
                                                        context={"field": key})
                            _finite(predicted, valid, key)
                            _finite(truth, valid, key)
                            errors = (predicted[valid] - truth[valid]).abs()
                            _finite(errors, slice(None), key)
                            errors = errors.cpu().tolist()
                            boundaries[key].extend(BoundaryObservation(total + row, target, prediction, error)
                                for row, target, prediction, error in zip(indices, truth[valid].cpu().tolist(),
                                                                        predicted[valid].cpu().tolist(), errors))
                total += len(chunk)
    return CohortObservations(total, {gene: tuple(rows) for gene, rows in classification.items()},
                              {key: tuple(rows) for key, rows in boundaries.items()})
