"""Allele recovery, thresholded sets, and explicitly observed equivalence-class comparisons."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from operator import index

from ..core.config import AlignAIRConfig
from ..errors import ConfigurationError, NumericalError, TrainingDataError
from .cohort import (ClassificationObservation, PredictionSetObservation, _threshold,
                     allele_observations, collect_alignment_observations)
from .statistics import _integer

_FIELDS = ("n", "nt_top1", "class_top1", "errors", "errors_within_class", "true_in_predicted_set", "set_size")


def empty_allele_counts() -> dict:
    return dict.fromkeys((*_FIELDS, "class_support", "class_error_support"), 0)


def _vocabulary(names, class_of):
    try:
        if isinstance(names, str):
            raise ValueError
        names = tuple(names)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("allele names must be a sequence") from exc
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ConfigurationError("allele names must be distinct nonempty strings")
    if class_of is not None and not isinstance(class_of, Mapping):
        raise ConfigurationError("equivalence classes must map allele names to class IDs")
    classes = None if class_of is None else tuple(class_of.get(name) for name in names)
    if classes is not None and any(value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0) for value in classes):
        raise ConfigurationError("equivalence class IDs must be nonnegative integers or unknown")
    return names, classes


def allele_observation_counts(classification, prediction_sets, names, class_of=None) -> dict:
    """Count aligned observations; absent equivalence labels never form a synthetic None class.

    Class metrics require mapped prediction and every truth member. Their support and the support
    of class-comparable allele errors are tracked separately. Allele recovery still uses every
    observed allele row, including rows whose equivalence labels are unavailable.
    """
    names, classes = _vocabulary(names, class_of)
    rows, sets = tuple(classification), tuple(prediction_sets)
    if len(rows) != len(sets):
        raise TrainingDataError("allele observations and predicted sets must have equal lengths")
    counts = empty_allele_counts()
    for row, predicted_set in zip(rows, sets):
        if (not isinstance(row, ClassificationObservation) or not isinstance(predicted_set, PredictionSetObservation)
                or row.record_index != predicted_set.record_index or not row.truth):
            raise TrainingDataError("allele observations and predicted sets must refer to the same observed records")
        try:
            ids = (*row.truth, row.prediction, *predicted_set.members)
            if any(isinstance(value, bool) or not 0 <= index(value) < len(names) for value in ids):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise TrainingDataError("allele observation index is outside the supplied vocabulary") from exc
        truth, predicted = set(row.truth), set(predicted_set.members)
        correct = row.prediction in truth
        counts["n"] += 1
        counts["nt_top1"] += correct
        counts["true_in_predicted_set"] += bool(truth & predicted)
        counts["set_size"] += len(predicted)
        if classes is not None:
            counts["errors"] += not correct
            labels = [classes[value] for value in row.truth]
            predicted_class = classes[row.prediction]
            if predicted_class is None or any(label is None for label in labels):
                continue
            class_correct = predicted_class in labels
            counts["class_support"] += 1
            counts["class_top1"] += class_correct
            counts["class_error_support"] += not correct
            counts["errors_within_class"] += not correct and class_correct
    return counts


def gene_allele_counts(probs, target, names, class_of, threshold: float = 0.5, *, supervision=None) -> dict:
    """Historical tensor counter, delegated to the collector's allele observation adapter."""
    if threshold is None:
        raise ConfigurationError("allele counters require a prediction-set threshold")
    rows, sets = allele_observations(probs, target, supervision=supervision, prediction_set_threshold=threshold)
    names, _ = _vocabulary(names, class_of)
    if probs.shape[0] and len(names) != probs.shape[1]:
        raise ConfigurationError("allele vocabulary length must match the probability width")
    return allele_observation_counts(rows, sets, names, class_of)


def merge_allele_counts(a, b) -> dict:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping) or a.keys() != b.keys():
        raise TrainingDataError("allele count ledgers must have matching fields")
    return {key: a[key] + b[key] for key in a}


def finalize_gene(counts: dict, has_class: bool) -> dict:
    """Normalize by observed support; historical ledgers without class-support fields remain readable."""
    if not isinstance(counts, Mapping) or not set(_FIELDS) <= counts.keys():
        raise TrainingDataError("allele count ledger is missing required fields")
    ledger = {key: counts[key] for key in _FIELDS}
    ledger["class_support"] = counts.get("class_support", counts["n"])
    ledger["class_error_support"] = counts.get("class_error_support", counts["errors"])
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in ledger.values()):
        raise TrainingDataError("allele counts must be nonnegative integers")
    ledger = {key: int(value) for key, value in ledger.items()}
    n, nc, ne = ledger["n"], ledger["class_support"], ledger["class_error_support"]
    if (ledger["nt_top1"] > n or ledger["errors"] > n or ledger["true_in_predicted_set"] > n
            or ledger["true_in_predicted_set"] > ledger["set_size"] or nc > n
            or ledger["class_top1"] > nc or ne > min(nc, ledger["errors"]) or ledger["errors_within_class"] > ne):
        raise TrainingDataError("allele count numerators exceed their observed support")
    result = {"n": n, "nt_top1": ledger["nt_top1"] / max(1, n),
              "true_in_predicted_set": ledger["true_in_predicted_set"] / max(1, n),
              "mean_pred_set_size": ledger["set_size"] / max(1, n)}
    if has_class:
        result.update(class_top1=ledger["class_top1"] / max(1, nc),
                      nt_error_rate=ledger["errors"] / max(1, n),
                      frac_errors_within_class=ledger["errors_within_class"] / max(1, ne),
                      class_support=nc, class_error_support=ne, unmapped_class_rows=n - nc)
    return result


def validate_allele_request(config, threshold, batch_size) -> int:
    """Validate allele semantics and sampling options before a study starts its source."""
    if not isinstance(config, AlignAIRConfig) or config.classification_mode != "multilabel":
        raise ConfigurationError("allele-cohort evaluation requires a multilabel AlignAIRConfig")
    _threshold(threshold)
    if threshold is None:
        raise ConfigurationError("allele-cohort evaluation requires a prediction-set threshold")
    return _integer(batch_size, "batch_size", 1)


def evaluate_allele_cohort(model, records, batch_adapter, config, reference_set, class_maps=None, *,
                           threshold: float = 0.5, batch_size: int = 256) -> dict:
    """Evaluate either input alphabet through a caller-selected allele batch adapter.

    Boundary and scalar metrics use their own observed denominators. Disabled metadata heads
    report null, while empty enabled heads retain zero-valued metrics with zero support.
    """
    batch_size = validate_allele_request(config, threshold, batch_size)
    class_maps = {} if class_maps is None else class_maps
    if not isinstance(class_maps, Mapping):
        raise ConfigurationError("class_maps must map gene names to equivalence maps")
    names = {}
    for spec in config.gene_specs:
        try:
            names[spec.name] = tuple(reference_set.gene(spec.name).names)
        except (AttributeError, KeyError, TypeError) as exc:
            raise ConfigurationError("reference must supply the configured allele vocabularies") from exc
        if len(names[spec.name]) != spec.allele_count:
            raise ConfigurationError("reference vocabulary size disagrees with the configured model head",
                                     context={"gene": spec.name})
        _vocabulary(names[spec.name], class_maps.get(spec.name.upper()))
    heads = (("mutation_rate",) + (() if getattr(config, "anchor_heads", False) else ("productive",))
             if config.use_meta_heads else ())         # anchor heads: productivity is derived, no head learns it
    observed = collect_alignment_observations(model, records, batch_adapter, config, batch_size=batch_size,
                                              prediction_set_threshold=threshold, scalar_heads=heads)
    alleles = {}
    for gene, rows in observed.classification.items():
        classes = class_maps.get(gene.upper())
        counts = allele_observation_counts(rows, observed.prediction_sets[gene], names[gene], classes)
        alleles[gene] = finalize_gene(counts, classes is not None)
    mutation = observed.scalars.get("mutation_rate")
    productive = observed.scalars.get("productive")
    if productive is not None:
        if any(row.truth not in (0, 1) for row in productive):
            raise TrainingDataError("productivity truth must be binary", context={"field": "productive"})
        if any(not 0 <= row.prediction <= 1 for row in productive):
            raise NumericalError("productivity predictions must be probabilities", context={"field": "productive"})
    return {"n": observed.record_count, "alleles": alleles,
            "segmentation_mae": {key: sum(row.absolute_error for row in rows) / max(1, len(rows))
                                 for key, rows in observed.boundaries.items()},
            "mutation_mae": (sum(row.absolute_error for row in mutation) / max(1, len(mutation))
                             if mutation is not None else None),
            "productive_acc": (sum((row.prediction > .5) == (row.truth > .5) for row in productive) / max(1, len(productive))
                               if productive is not None else None),
            "support": observed.support}
