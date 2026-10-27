"""Protein-study frozen-cohort inference and coverage/ambiguity strata.

Reusable statistics and report calculations live in alignair.evaluate. This study adapter retains
its V/J exact-class policy, categorical batch construction and protein-specific strata.
"""
from __future__ import annotations

from collections.abc import Sequence

from ..core.config import AlignAIRConfig
from ..data.adapters import CategoricalBatchAdapter
from ..errors import ConfigurationError
from ..evaluate.cohort import collect_alignment_observations
from ..evaluate.reports import (boundary_error_report, classification_report,
                                constant_median_mae, majority_class_baseline,
                                macro_accuracy as _macro_accuracy)
from ..evaluate.statistics import bootstrap_mean_ci, paired_diff_ci, wilson_interval


def evaluate_frozen(model, samples, cfg, class_maps, *, batch_size: int = 128,
                    device: str = "cpu") -> dict:
    """Report supervised V/J observations with per-field support and protein-specific strata."""
    if not isinstance(cfg, AlignAIRConfig) or cfg.classification_mode != "categorical":
        raise ConfigurationError("the protein frozen report requires categorical model heads")
    if not isinstance(samples, Sequence):
        raise ConfigurationError("protein strata require an indexable frozen sample sequence",
                                 context={"field": "samples"})
    observations = collect_alignment_observations(
        model, samples, CategoricalBatchAdapter.from_config(cfg, device=device), cfg,
        batch_size=batch_size, genes=("v", "j"),
    )
    res: dict = {"n": observations.record_count, "classification": {}, "boundaries": {},
                 "stratified": {}, "support": observations.support}
    for gene, rows in observations.classification.items():
        if any(len(row.truth) != 1 for row in rows):
            raise ConfigurationError(
                "exact-class frozen reports require singleton truth",
                context={"gene": gene},
                hint="Use collect_alignment_observations and membership correctness for compatible-label reports.",
            )
        res["classification"][gene] = classification_report(
            [row.truth[0] for row in rows], [row.prediction for row in rows])
    for key, rows in observations.boundaries.items():
        res["boundaries"][key] = boundary_error_report(
            [row.truth for row in rows], [row.absolute_error for row in rows])

    # V accuracy stratified by observed coverage and by protein-class ambiguity size (DIAGNOSTIC)
    cov_buckets, amb_buckets = {}, {}
    vmap = class_maps["V"]
    for row in observations.classification["v"]:
        s, t, p = samples[row.record_index], row.truth[0], row.prediction
        cov = s["v_end"] - s["v_start"]
        cb = "<40" if cov < 40 else ("40-79" if cov < 80 else ">=80")
        hit, n = cov_buckets.get(cb, (0, 0))
        cov_buckets[cb] = (hit + int(t == p), n + 1)
        amb = len(vmap.ambiguity_set(t))
        ab = str(amb) if amb < 3 else ">=3"
        hit, n = amb_buckets.get(ab, (0, 0))
        amb_buckets[ab] = (hit + int(t == p), n + 1)
    res["stratified"]["v_acc_by_coverage"] = {
        k: {"acc": h / n, "n": n, "ci95": wilson_interval(h, n)}
        for k, (h, n) in sorted(cov_buckets.items())}
    res["stratified"]["v_acc_by_ambiguity_size"] = {
        k: {"acc": h / n, "n": n, "ci95": wilson_interval(h, n)}
        for k, (h, n) in sorted(amb_buckets.items())}
    return res
