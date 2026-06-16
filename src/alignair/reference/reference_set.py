"""ReferenceSet: union 1..N GenAIRR dataconfigs into per-gene allele references."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch


@dataclass
class GeneReference:
    names: List[str]          # ordered allele names
    sequences: List[str]      # germline nucleotide seqs (uppercased), aligned with names
    index: Dict[str, int]     # name -> row index

    def __len__(self) -> int:
        return len(self.names)


class ReferenceSet:
    """Per-gene union allele references built from one or more GenAIRR DataConfigs."""

    def __init__(self, genes: Dict[str, GeneReference], has_d: bool):
        self.genes = genes
        self.has_d = has_d

    def gene(self, g: str) -> GeneReference:
        return self.genes[g.upper()]

    @classmethod
    def from_dataconfigs(cls, *dataconfigs) -> "ReferenceSet":
        has_d = any(dc.metadata.has_d for dc in dataconfigs)
        wanted = ["v", "j"] + (["d"] if has_d else [])
        genes: Dict[str, GeneReference] = {}
        for g in wanted:
            names: List[str] = []
            sequences: List[str] = []
            index: Dict[str, int] = {}
            for dc in dataconfigs:
                if g == "d" and not dc.metadata.has_d:
                    continue
                for allele in dc.allele_list(g):
                    if allele.name in index:
                        continue
                    index[allele.name] = len(names)
                    names.append(allele.name)
                    sequences.append(allele.ungapped_seq.upper())
            genes[g.upper()] = GeneReference(names, sequences, index)
        return cls(genes, has_d)

    def genotype_mask(self, gene: str, allowed_names: Iterable[str]) -> torch.Tensor:
        ref = self.gene(gene)
        allowed = set(allowed_names)
        return torch.tensor([n in allowed for n in ref.names], dtype=torch.bool)
