"""Coarse filter (retriever) — shortlists V candidates for the prompt.

The full reference is too big for the context, so a cheap, recall-oriented filter picks the top-k V
germlines by shared-k-mer overlap with the query (BLAST-like, no learning). D and J are small enough
to keep in full, so they don't need this. Deliberately isolated behind `fit`/`shortlist` so it can be
swapped for a better retriever (learned, minimizer-based, hierarchical) without touching the rest.

Recall is the ceiling on novel-allele accuracy: if the true allele isn't shortlisted, the LM cannot
copy it — so during training the true allele is force-included.
"""
from __future__ import annotations


def _kmers(seq: str, k: int) -> set:
    s = seq.upper()
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


class KmerFilter:
    """v1 coarse filter: rank candidates by count of k-mers shared with the query."""

    def __init__(self, k: int = 11):
        self.k = k
        self.cand_kmers: list[set] = []

    def fit(self, candidates: list[str]) -> "KmerFilter":
        """Precompute candidate k-mer sets once per reference (candidates in reference order)."""
        self.cand_kmers = [_kmers(c, self.k) for c in candidates]
        return self

    def shortlist(self, query: str, k: int, force_include: int | None = None) -> list[int]:
        """Return indices of the top-k candidates by shared-k-mer overlap (ties broken by index).
        `force_include` guarantees an index is present (the training positive)."""
        qk = _kmers(query, self.k)
        scores = [len(qk & ck) for ck in self.cand_kmers]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        idx = order[: min(k, len(order))]
        if force_include is not None and force_include not in idx:
            idx = idx[: max(k - 1, 0)] + [force_include]
        return idx
