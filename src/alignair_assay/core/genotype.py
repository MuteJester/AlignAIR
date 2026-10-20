"""Donor genotypes as plain data: what a simulated donor carries, readable by truth, metrics and scoring.

The sampler that creates donors, and their GenAIRR form, live in ``suites.donors``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NovelAllele:
    """An allele absent from the reference, derived from reference allele ``base`` by substitutions only:
    ``sequence`` (uppercase germline, same length as the base) differs from the base at ``snps`` (0-based
    germline positions, sorted) and nowhere else. ``name`` belongs to the base's gene (text before ``*``).
    ValueError, naming the allele, when ``snps`` is empty, unsorted, repeats a position or leaves the sequence;
    ``check_base`` checks the sequence against the base's."""

    name: str
    base: str
    sequence: str
    snps: tuple[int, ...]

    def __post_init__(self) -> None:
        snps = list(self.snps)
        if not snps or snps != sorted(set(snps)) or not 0 <= snps[0] <= snps[-1] < len(self.sequence):
            raise ValueError(f"novel allele {self.name}: snps must be distinct sorted positions in "
                             f"[0, {len(self.sequence)}), got {self.snps!r}")

    def check_base(self, base_sequence: str) -> None:
        """ValueError, naming the allele, unless ``sequence`` has the length of ``base_sequence`` (the base's
        germline) and differs from it exactly at ``snps``."""
        if len(self.sequence) != len(base_sequence):
            raise ValueError(f"novel allele {self.name}: {len(self.sequence)} nt, but its base {self.base} has "
                             f"{len(base_sequence)}")
        differing = tuple(p for p, (a, b) in enumerate(zip(self.sequence, base_sequence.upper())) if a != b)
        if differing != tuple(self.snps):
            raise ValueError(f"novel allele {self.name}: differs from its base {self.base} at {list(differing)}, "
                             f"not at its snps {list(self.snps)}")


@dataclass(frozen=True)
class DonorGenotype:
    """What donor ``subject_id`` carries: ``carried`` are allele names (novel names included) and ``deleted`` the
    homozygously deleted gene names."""

    subject_id: str
    carried: frozenset[str]
    deleted: frozenset[str]
