"""Protein-allele study adapter; counts and evaluation mechanics have shared owners."""
from __future__ import annotations

from ..data.adapters import ProteinAlleleBatchAdapter
from ..evaluate.alleles import (evaluate_allele_cohort, finalize_gene, gene_allele_counts,
                               merge_allele_counts as _merge)


def evaluate_aa(model, records, reference_set, cfg, class_maps, *, device="cpu",
                threshold: float = 0.5, batch_size: int = 256) -> dict:
    """Evaluate translated reads through the shared allele-cohort runner."""
    return evaluate_allele_cohort(model, records, ProteinAlleleBatchAdapter(reference_set, cfg, device),
                                  cfg, reference_set, class_maps, threshold=threshold, batch_size=batch_size)
