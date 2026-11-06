"""Genotypes: genotype files (``files``), a genotype as the chip's bank (``bank``), the lean frequency rule that infers
a donor's V genotype from AIRR rows (``lean``), and the AIRR GenotypeSet it is written as (``airr``)."""
from .bank import genotype_bank
from .files import UnknownAlleleError, load_genotype

__all__ = ["genotype_bank", "load_genotype", "UnknownAlleleError"]
