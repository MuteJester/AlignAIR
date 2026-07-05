"""K-mer seed shortlisting: cut the fine-scoring candidate set from all-of-R down to top-k.

Classical BLAST-style seeding on raw sequence (NOT neural embeddings, whose pooled vectors
retrieve at chance). Reference-agnostic, CPU-cheap (inverted index), and high-recall: the true
allele lands in the top-16 (top-32 under heavy SHM). Smaller k is more SHM-robust.
"""
from collections import defaultdict
from collections import Counter


def kmers(seq: str, k: int) -> set:
    s = seq.upper()
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else set()


class KmerIndex:
    """Inverted k-mer index over one gene's unique sequences for fast shortlist + strand pick."""

    def __init__(self, unique_seqs, k: int = 9):
        self.k = k
        self.n = len(unique_seqs)
        self.inv = defaultdict(list)
        self.pos_inv = defaultdict(list)
        for ci, s in enumerate(unique_seqs):
            seen = set()
            for pos in range(max(len(s) - k + 1, 0)):
                km = s[pos:pos + k].upper()
                self.pos_inv[km].append((ci, pos))
                if km not in seen:
                    self.inv[km].append(ci)   # each candidate listed once per k-mer
                    seen.add(km)

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

    def anchor_offsets(self, seq: str, candidate_indices) -> list[int]:
        """Best read_start - candidate_start offset per candidate from shared k-mer hits.

        The banded scorer uses j ~= i + offset, where i is a candidate allele position and
        j is a read position. If a candidate has no k-mer hit in the read, fall back to 0.
        """
        wanted = {int(i) for i in candidate_indices}
        hist = {int(i): Counter() for i in candidate_indices}
        s = seq.upper()
        for rpos in range(max(len(s) - self.k + 1, 0)):
            km = s[rpos:rpos + self.k]
            for ci, cpos in self.pos_inv.get(km, ()):
                if ci in wanted:
                    hist[ci][rpos - cpos] += 1
        out = []
        for ci in candidate_indices:
            h = hist[int(ci)]
            out.append(h.most_common(1)[0][0] if h else 0)
        return out
