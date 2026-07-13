"""Experimental: infer an individual's IG genotype from a repertoire, grounded in the model's
predictions and its allele-prototype geometry. See
docs/superpowers/specs/2026-07-10-alignair-genotype-inference-design.md.
"""
from .constraint import (NovelAlleleUnsupportedError, adjust_for_genotype, genotype_allowed_mask,
                         load_genotype)
from .geometry import LeakageModel, allele_prototypes, prototype_cosine, residual_support
from .infer import GenotypeParams, GenotypeResult, decide_gene_calls, infer_genotype

# `study` (constraint benchmark) is a heavy submodule (pulls api/evaluate) — import on demand as
# `alignair.genotype.study`, not eagerly here.
__all__ = ["allele_prototypes", "prototype_cosine", "LeakageModel", "residual_support",
           "infer_genotype", "decide_gene_calls", "GenotypeResult", "GenotypeParams",
           "adjust_for_genotype", "genotype_allowed_mask", "load_genotype",
           "NovelAlleleUnsupportedError"]
