"""Experimental: infer an individual's IG genotype from a repertoire, grounded in the model's
predictions and its allele-prototype geometry. See
docs/superpowers/specs/2026-07-10-alignair-genotype-inference-design.md.
"""
from .geometry import LeakageModel, allele_prototypes, prototype_cosine, residual_support
from .infer import GenotypeResult, infer_genotype

__all__ = ["allele_prototypes", "prototype_cosine", "LeakageModel", "residual_support",
           "infer_genotype", "GenotypeResult"]
