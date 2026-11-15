"""Whether a read holds a V: a read without one (a D-J rearrangement, a read that starts inside the junction) gets no V
call, as an absent D gets no D call (``core.d_presence``).

A read holds no V when both hold:

- it holds no exact ``V_KMER``-mer of any V allele of the reference (the AlignAIR assay's real-data truth calls such a
  read one without a V: on four donors 3 of 157,188 reads with an IgBLAST V of at least 100 nt hold none);
- the V the germline reader placed is no V either: under ``V_MIN_SPAN`` read bases (the assay asks a call of no
  shorter V) or under ``V_MIN_IDENTITY`` identity along its CIGAR. A heavily mutated V can keep no exact k-mer (a read
  that starts in FR3, at high SHM) and keeps its call by its span and identity.
"""
from __future__ import annotations

import functools

from .cigar import ALIGNED, cigar_ops, matched_bases

V_KMER = 15
V_MIN_SPAN = 20
V_MIN_IDENTITY = 0.7
V_PRESENCE = "v_presence"                         # the record field a gated read carries
V_ABSENT = "absent"                               # its value


@functools.lru_cache(maxsize=8)
def _kmers(sequences: tuple) -> frozenset:
    return frozenset(seq[i:i + V_KMER] for seq in sequences for i in range(len(seq) - V_KMER + 1))


def _holds_kmer(sequence: str, kmers: frozenset) -> bool:
    sequence = sequence.upper()
    return any(sequence[i:i + V_KMER] in kmers for i in range(len(sequence) - V_KMER + 1))


def _identity(sequence: str, germline: str, aln) -> float | None:
    """The share of the alignment's columns (aligned bases, insertions and deletions) whose read and germline bases
    are equal; None for an alignment without a column."""
    read, germ = sequence[aln.seq_start:aln.seq_end].upper(), germline[aln.germ_start:aln.germ_end]
    columns = sum(count for count, op in cigar_ops(aln.cigar) if op in ALIGNED or op in "ID")
    return matched_bases(read, germ, aln.cigar) / columns if columns else None


def _credible(sequence: str, aln, germlines: dict) -> bool:
    if aln.seq_end - aln.seq_start < V_MIN_SPAN:
        return False
    identity = _identity(sequence, germlines.get(aln.allele, ""), aln)
    return identity is not None and identity >= V_MIN_IDENTITY


def v_absent(sequences, alignments, reference) -> list:
    """Per read (forward frame), whether it holds no V: its V alignment is no V, and it holds no exact ``V_KMER``-mer
    of any V allele of ``reference``. A read without a V call (alignment ``None``) has no V to drop."""
    germlines = kmers = None
    out = []
    for sequence, aln in zip(sequences, alignments):
        if aln is None:
            out.append(False)
            continue
        if germlines is None:
            gene = reference.gene("V")
            germlines = {name: seq.upper() for name, seq in zip(gene.names, gene.sequences)}
        if _credible(sequence, aln, germlines):
            out.append(False)
            continue
        if kmers is None:
            kmers = _kmers(tuple(germlines.values()))
        out.append(not _holds_kmer(sequence, kmers))
    return out
