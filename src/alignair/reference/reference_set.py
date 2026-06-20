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

    @classmethod
    def from_genotype(cls, genes: Dict[str, Dict[str, str]]) -> "ReferenceSet":
        """Build a reference from a genotype mapping {gene_type: {allele_name: dna_seq}}.

        gene_type is V/J (+ D for heavy chains; omit D for light). Allele names need
        NOT be from the training reference — NOVEL alleles are just rows the encoder
        will embed at predict time, so the model conditions on whatever it is handed.
        """
        upper = {k.upper(): v for k, v in genes.items()}
        has_d = bool(upper.get("D"))
        wanted = ["V", "J"] + (["D"] if has_d else [])
        out: Dict[str, GeneReference] = {}
        for g in wanted:
            names: List[str] = []
            sequences: List[str] = []
            index: Dict[str, int] = {}
            for name, seq in upper.get(g, {}).items():
                if name in index:
                    continue
                index[name] = len(names)
                names.append(name)
                sequences.append(str(seq).upper().replace("-", "").replace(".", ""))
            out[g] = GeneReference(names, sequences, index)
        return cls(out, has_d)

    @classmethod
    def from_yaml(cls, path: str) -> "ReferenceSet":
        """Load a genotype YAML: top-level keys v/d/j, each {allele_name: dna_seq}."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_genotype(data)

    def to_yaml(self, path: str) -> None:
        import yaml
        data = {g.lower(): dict(zip(ref.names, ref.sequences))
                for g, ref in self.genes.items()}
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def genotype_mask(self, gene: str, allowed_names: Iterable[str]) -> torch.Tensor:
        ref = self.gene(gene)
        allowed = set(allowed_names)
        return torch.tensor([n in allowed for n in ref.names], dtype=torch.bool)
