"""Exact-class cohort reports shared by nucleotide and protein input models."""
from __future__ import annotations

from collections.abc import Sequence

from ..core.config import AlignAIRConfig
from ..data.batching import BatchAdapter
from ..errors import ConfigurationError, error_context
from .cohort import _selected_gene_counts, collect_alignment_observations
from .reports import boundary_error_report, classification_report
from .statistics import wilson_interval
from .strata import ClassificationStratum


def evaluate_categorical_cohort(model, records, batch_adapter: BatchAdapter, config: AlignAIRConfig, *,
                                 batch_size: int = 128, genes=None, strata=()) -> dict:
    """Report singleton class truth and boundaries through a caller-selected batch adapter.

    By default all configured genes are reported without strata and records may be an iterator.
    Strata require an indexable sequence to join supervised rows to their original records.
    Class masks and boundary masks keep their independent support. Compatible multi-class truth
    requires a membership report, not this exact-class summary.
    """
    if not isinstance(config, AlignAIRConfig) or config.classification_mode != "categorical":
        raise ConfigurationError("categorical-cohort evaluation requires a categorical AlignAIRConfig")
    counts = _selected_gene_counts(config, genes)
    try:
        strata = tuple(strata)
    except TypeError as exc:
        raise ConfigurationError("strata must be an iterable of ClassificationStratum policies") from exc
    if any(not isinstance(policy, ClassificationStratum) for policy in strata):
        raise ConfigurationError("strata must contain ClassificationStratum policies")
    if len({policy.name for policy in strata}) != len(strata):
        raise ConfigurationError("classification stratum names must be unique")
    if any(policy.gene not in counts for policy in strata):
        raise ConfigurationError("classification strata must select a reported gene")
    if strata and not isinstance(records, Sequence):
        raise ConfigurationError("classification strata require an indexable frozen record sequence",
                                 context={"field": "records"})
    observations = collect_alignment_observations(
        model, records, batch_adapter, config, batch_size=batch_size, genes=tuple(counts))
    report = {"n": observations.record_count, "classification": {}, "boundaries": {},
              "stratified": {}, "support": observations.support}
    for gene, rows in observations.classification.items():
        if any(len(row.truth) != 1 for row in rows):
            raise ConfigurationError(
                "exact-class cohort reports require singleton truth", context={"gene": gene},
                hint="Use collect_alignment_observations and membership correctness for compatible-label reports.")
        report["classification"][gene] = classification_report(
            [row.truth[0] for row in rows], [row.prediction for row in rows])
    for key, rows in observations.boundaries.items():
        report["boundaries"][key] = boundary_error_report(
            [row.truth for row in rows], [row.absolute_error for row in rows])
    for policy in strata:
        buckets = {}
        for row in observations.classification[policy.gene]:
            with error_context(stage="classification_stratification", stratum=policy.name,
                               gene=policy.gene, record_index=row.record_index):
                label = policy.group(records[row.record_index], row.truth[0])
                if not isinstance(label, str) or not label.strip():
                    raise ConfigurationError("classification stratum labels must be nonempty strings")
            hits, count = buckets.get(label, (0, 0))
            buckets[label] = hits + int(row.correct), count + 1
        report["stratified"][policy.name] = {
            label: {"acc": hits / count, "n": count, "ci95": wilson_interval(hits, count)}
            for label, (hits, count) in sorted(buckets.items())}
    return report
