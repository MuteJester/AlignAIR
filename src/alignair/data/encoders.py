"""Allele multi-hot and chain-type one-hot encoders (port of legacy encoders)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class GeneEncoding:
    allele_to_index: dict
    index_to_allele: dict
    count: int


class AlleleEncoder:
    """Multi-hot encode sets of allele calls per gene type."""

    def __init__(self):
        self.gene_encodings: dict[str, GeneEncoding] = {}

    def register_gene(self, gene_type: str, allele_list, sort: bool = True,
                      allow_overwrite: bool = False) -> None:
        if gene_type in self.gene_encodings and not allow_overwrite:
            raise ValueError(f"Gene '{gene_type}' already registered")
        alleles = sorted(allele_list) if sort else list(allele_list)
        a2i = {a: i for i, a in enumerate(alleles)}
        i2a = {i: a for a, i in a2i.items()}
        self.gene_encodings[gene_type] = GeneEncoding(a2i, i2a, len(alleles))

    def count(self, gene_type: str) -> int:
        return self.gene_encodings[gene_type].count

    def encode(self, gene_type: str, allele_sets) -> np.ndarray:
        enc = self.gene_encodings[gene_type]
        rows = []
        for sample in allele_sets:
            ohe = np.zeros(enc.count, dtype=np.float32)
            for allele in sample:
                idx = enc.allele_to_index.get(allele)
                if idx is not None:
                    ohe[idx] = 1.0
            rows.append(ohe)
        return np.vstack(rows)


class ChainTypeEncoder:
    """One-hot encode chain-type labels in a fixed order."""

    def __init__(self, chain_types):
        self.chain_types = [str(c) for c in chain_types]
        self._index = {c: i for i, c in enumerate(self.chain_types)}

    @property
    def count(self) -> int:
        return len(self.chain_types)

    def encode(self, labels) -> np.ndarray:
        rows = []
        for label in labels:
            ohe = np.zeros(self.count, dtype=np.float32)
            idx = self._index.get(str(label))
            if idx is not None:
                ohe[idx] = 1.0
            rows.append(ohe)
        return np.vstack(rows)
