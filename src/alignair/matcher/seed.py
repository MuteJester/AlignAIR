"""K-mer seed shortlisting: cut the fine-scoring candidate set from all-of-R down to top-k.

Classical BLAST-style seeding on raw sequence (NOT neural embeddings, whose pooled vectors
retrieve at chance). Reference-agnostic, CPU-cheap (inverted index), and high-recall: the true
allele lands in the top-16 (top-32 under heavy SHM). Smaller k is more SHM-robust.
"""
from collections import defaultdict


def kmers(seq: str, k: int) -> set:
    s = seq.upper()
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else set()


class KmerIndex:
    """Inverted k-mer index over one gene's unique sequences for fast shortlist + strand pick."""

    def __init__(self, unique_seqs, k: int = 9):
        self.k = k
        self.n = len(unique_seqs)
        self.inv = defaultdict(list)
        for ci, s in enumerate(unique_seqs):
            for km in kmers(s, k):
                self.inv[km].append(ci)   # each candidate listed once per k-mer (kmers() dedups)

    def hit_counts(self, seq: str):
        """counts[i] = number of the read's unique k-mers present in candidate i."""
        counts = [0] * self.n
        inv = self.inv
        for km in kmers(seq, self.k):
            for ci in inv.get(km, ()):
                counts[ci] += 1
        return counts

    def shortlist(self, seq: str, topk: int):
        """Return (candidate indices sorted by descending k-mer hits, capped at topk; counts)."""
        counts = self.hit_counts(seq)
        order = sorted(range(self.n), key=lambda i: counts[i], reverse=True)
        return order[:min(topk, self.n)], counts

    def best_hits(self, seq: str) -> int:
        """Max k-mer hit count over candidates — a cheap orientation/quality signal."""
        c = self.hit_counts(seq)
        return max(c) if c else 0
