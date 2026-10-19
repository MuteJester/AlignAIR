"""Donor genotypes as plain data: what a simulated donor carries, readable by truth, metrics and scoring.

The sampler that creates donors, and their GenAIRR form, live in ``suites.donors``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NovelAllele:
    """An allele absent from the reference, derived from reference allele ``base`` by substitutions only:
    ``sequence`` (uppercase germline, same length as the base) differs from the base at ``snps`` (0-based
    germline positions, sorted). ``name`` belongs to the base's gene (text before ``*``)."""

    name: str
    base: str
    sequence: str
    snps: tuple[int, ...]


@dataclass(frozen=True)
class DonorGenotype:
    """What donor ``subject_id`` carries: ``carried`` are allele names (novel names included) and ``deleted`` the
    homozygously deleted gene names."""

    subject_id: str
    carried: frozenset[str]
    deleted: frozenset[str]
