from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...core import GENES


class _Accumulator:
    def __init__(self) -> None:
        self.n_cases = 0
        self.global_sums: dict[str, float] = defaultdict(float)
        self.global_counts: dict[str, int] = defaultdict(int)
        self.gene_sums: dict[str, dict[str, float]] = {g: defaultdict(float) for g in GENES}
        self.gene_counts: dict[str, dict[str, int]] = {g: defaultdict(int) for g in GENES}

    def update(self, scored: dict[str, Any]) -> None:
        self.n_cases += 1
        for k, v in scored.get("global", {}).items():
            self.global_sums[k] += float(v)
            self.global_counts[k] += 1
        for gene, vals in scored.get("genes", {}).items():
            if gene not in self.gene_sums:
                self.gene_sums[gene] = defaultdict(float)
                self.gene_counts[gene] = defaultdict(int)
            for k, v in vals.items():
                self.gene_sums[gene][k] += float(v)
                self.gene_counts[gene][k] += 1

    @staticmethod
    def _finalize(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float | None]:
        return {k: (sums[k] / counts[k] if counts[k] else None) for k in sorted(sums)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "global": self._finalize(self.global_sums, self.global_counts),
            "genes": {
                g: self._finalize(self.gene_sums.get(g, {}), self.gene_counts.get(g, {}))
                for g in GENES
            },
        }
