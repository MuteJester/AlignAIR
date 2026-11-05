"""Genotypes: the constraint a known genotype puts on prediction (``constraint``), the lean frequency rule that
infers a donor's V genotype from AIRR rows (``lean``), and the AIRR GenotypeSet it is written as (``airr``)."""
from .constraint import (NovelAlleleUnsupportedError, adjust_for_genotype, genotype_allowed_mask,
                         load_genotype)

__all__ = ["adjust_for_genotype", "genotype_allowed_mask", "load_genotype", "NovelAlleleUnsupportedError"]
