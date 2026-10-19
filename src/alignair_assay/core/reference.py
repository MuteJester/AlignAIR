"""Germline reference used by the assay: allele sequences per gene plus conserved anchors.

This is the only assay module that imports ``alignair`` (``Reference.from_dataconfig``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from alignair_assay.core.cases import read_fasta

# The GenAIRR dataconfig every command uses unless ``--reference`` names another.
DEFAULT_REFERENCE = "HUMAN_IGH_OGRDB"


def dataconfig(name: str):
    """The GenAIRR DataConfig ``name`` (an attribute of ``GenAIRR.data``, e.g. ``HUMAN_IGH_OGRDB``); ValueError for
    an unknown name. Suites simulate from this object and the truth reference is built from it."""
    import GenAIRR.data as gdata

    config = getattr(gdata, name, None)
    if config is None:
        raise ValueError(f"unknown GenAIRR dataconfig {name!r}")
    return config


def gene_name(allele: str) -> str:
    """Gene part of an allele name (text before ``*``)."""
    return allele.split("*", 1)[0]


@dataclass(frozen=True)
class Reference:
    """The germline alleles every truth and score is built against, keyed by gene letter and allele name, with the
    conserved junction anchors (0-based germline positions)."""

    # gene letter ("v" | "d" | "j") -> allele name -> uppercase germline sequence
    alleles: dict[str, dict[str, str]]
    # allele name -> 0-based germline anchor (V Cys codon start, J Trp/Phe codon start)
    anchors: dict[str, int]

    def seq(self, name: str) -> str:
        """Germline sequence of allele ``name`` (KeyError when it is not in the reference)."""
        return self.alleles[self.gene_letter(name)][name]

    def gene_letter(self, name: str) -> str:
        """``"v"``, ``"d"`` or ``"j"``: the gene table holding allele ``name`` (KeyError when none does)."""
        for letter, table in self.alleles.items():
            if name in table:
                return letter
        raise KeyError(name)

    def has(self, name: str) -> bool:
        """True when allele ``name`` is in the reference."""
        return any(name in table for table in self.alleles.values())

    def digest(self) -> str:
        """SHA-256 over every allele's gene letter, name and sequence, in sorted order (anchors excluded)."""
        h = hashlib.sha256()
        for letter in sorted(self.alleles):
            for name in sorted(self.alleles[letter]):
                h.update(f"{letter}\t{name}\t{self.alleles[letter][name]}\n".encode())
        return h.hexdigest()

    @classmethod
    def from_reference_set(cls, rs) -> "Reference":
        """Copy an ``alignair`` ReferenceSet: uppercase sequences per gene (D only when the locus has one) and the
        V/J anchors."""
        alleles: dict[str, dict[str, str]] = {}
        anchors: dict[str, int] = {}
        for letter in ("V", "D", "J"):
            if letter == "D" and not rs.has_d:
                continue
            gene = rs.gene(letter)
            alleles[letter.lower()] = dict(zip(gene.names, (s.upper() for s in gene.sequences)))
            if gene.anchors:
                anchors.update({k: int(v) for k, v in gene.anchors.items()})
        return cls(alleles=alleles, anchors=anchors)

    @classmethod
    def from_fasta(cls, path) -> "Reference":
        """The reference a FASTA file holds (for example one a tool exported): one allele per record, named by
        the header's first word, with its sequence uppercased. The gene letter is the lower-cased fourth
        character of the name (``IGHV1-2*02`` -> ``v``). Anchors are not in a FASTA file, so they are ``{}``.
        A name whose letter is not ``v``, ``d`` or ``j``, or that appears twice, raises ValueError."""
        alleles: dict[str, dict[str, str]] = {}
        for name, sequence in read_fasta(path):
            letter = name[3:4].lower()
            if letter not in ("v", "d", "j"):
                raise ValueError(f"{path}: allele {name!r} has no V, D or J gene letter as its fourth character")
            if any(name in table for table in alleles.values()):
                raise ValueError(f"{path}: allele {name!r} appears twice")
            alleles.setdefault(letter, {})[name] = sequence.upper()
        return cls(alleles=alleles, anchors={})

    @classmethod
    def from_dataconfig(cls, name: str) -> "Reference":
        """The reference of GenAIRR dataconfig ``name`` (see ``dataconfig``); ValueError for an unknown name."""
        from alignair.reference.reference_set import ReferenceSet

        return cls.from_reference_set(ReferenceSet.from_dataconfigs(dataconfig(name)))
