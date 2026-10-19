"""Germline reference used by the assay: allele sequences per gene plus conserved anchors."""
from __future__ import annotations

from dataclasses import dataclass


def gene_name(allele: str) -> str:
    """Gene part of an allele name (text before ``*``)."""
    return allele.split("*", 1)[0]


@dataclass(frozen=True)
class Reference:
    # gene letter ("v" | "d" | "j") -> allele name -> uppercase germline sequence
    alleles: dict[str, dict[str, str]]
    # allele name -> 0-based germline anchor (V Cys codon start, J Trp/Phe codon start)
    anchors: dict[str, int]

    def seq(self, name: str) -> str:
        return self.alleles[self.gene_letter(name)][name]

    def gene_letter(self, name: str) -> str:
        for letter, table in self.alleles.items():
            if name in table:
                return letter
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return any(name in table for table in self.alleles.values())

    @classmethod
    def from_reference_set(cls, rs) -> "Reference":
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
    def from_dataconfig(cls, name: str) -> "Reference":
        import GenAIRR.data as gdata

        from alignair.reference.reference_set import ReferenceSet

        return cls.from_reference_set(ReferenceSet.from_dataconfigs(getattr(gdata, name)))
