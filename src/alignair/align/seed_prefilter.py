from __future__ import annotations
from collections import defaultdict


def _kmers(seq: str, k: int):
    return (seq[i:i + k] for i in range(len(seq) - k + 1))


class SeedPrefilter:
    """Non-learned k-mer admission: rank candidate alleles by shared-k-mer count with a read
    segment, independent of the neural encoder. Lets a divergent/novel allele enter the
    alignment pool even when pooled-cosine retrieval misranks it."""

    def __init__(self, reference_set, k: int = 11):
        self.k = k
        self._index = {}          # gene -> {kmer: set(allele_idx)}
        for G in ("V", "D", "J"):
            try:
                gene = reference_set.gene(G)
            except Exception:
                continue
            idx = defaultdict(set)
            for ai, seq in enumerate(gene.sequences):
                for km in _kmers(seq.upper(), k):
                    idx[km].add(ai)
            self._index[G] = idx

    def candidates(self, segment: str, gene: str, m: int,
                   allowed: set[int] | None = None) -> list[int]:
        idx = self._index.get(gene.upper())
        if idx is None or len(segment) < self.k:
            return []
        counts = defaultdict(int)
        for km in _kmers(segment.upper(), self.k):
            for ai in idx.get(km, ()):
                if allowed is None or ai in allowed:
                    counts[ai] += 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [ai for ai, _ in ranked[:m]]
