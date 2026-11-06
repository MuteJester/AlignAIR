"""The V alleles a read cannot tell from its call: the record's ``v_call`` lists them after the call, as IgBLAST lists
equally scoring alleles, so a genotype step downstream (TIgGER, Change-O) can choose among them.

An allele ties with the call when it matches at least as many of the read's V bases as the call does: the read bases
the call's alignment puts on germline bases (its CIGAR's M columns), each compared with the allele's base at the same
IMGT column (the IMGT gapped forms put alleles of other lengths and with internal gaps in one frame; a reference
without them compares germline positions). An allele without a base at a column (a gap, or past its end) matches no
read base there, and an inserted read base is compared with no allele. Only an allele of the bound reference ties (a
genotype is the bank, so its alleles are the only candidates). The ties follow the call by matches, then in reference
order.
"""
from __future__ import annotations

import functools
import re

import numpy as np

_CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")
_NO_BASE = 0                                      # the code of a column an allele holds no base at


@functools.lru_cache(maxsize=4)
def _table(gapped: tuple) -> tuple:
    """``(bases, columns)``: the alleles' base codes over the IMGT columns, (alleles, columns) uint8 with ``_NO_BASE``
    where an allele holds none, and per allele the IMGT column of each of its germline positions."""
    width = max((len(form) for form in gapped), default=0)
    bases = np.full((len(gapped), width), _NO_BASE, dtype=np.uint8)
    columns = []
    for row, form in enumerate(gapped):
        codes = np.frombuffer(form.upper().encode(), dtype=np.uint8)
        held = codes != ord(".")
        bases[row, :len(codes)] = np.where(held, codes, _NO_BASE)
        columns.append(np.flatnonzero(held))
    return bases, tuple(columns)


def _aligned(aln) -> tuple:
    """The read and germline positions of the alignment's M columns."""
    reads, germs = [], []
    r, g = aln.seq_start, aln.germ_start
    for count, op in _CIGAR.findall(aln.cigar or ""):
        count = int(count)
        if op in "M=X":
            reads.extend(range(r, r + count))
            germs.extend(range(g, g + count))
            r, g = r + count, g + count
        elif op == "I":
            r += count
        elif op == "D":
            g += count
    return np.array(reads, dtype=np.int64), np.array(germs, dtype=np.int64)


def tied_alleles(sequences, alignments, reference) -> list:
    """Per read (forward frame), its V call and the V alleles of the bound bank tied with it, the call first (a tuple
    of names), or ``()`` for a read without a V alignment."""
    names = None
    out = []
    for sequence, aln in zip(sequences, alignments):
        if aln is None:
            out.append(())
            continue
        if names is None:
            gene = reference.gene("V")
            names, forms = list(gene.names), gene.gapped or {}
            bases, columns = _table(tuple(forms.get(name) or seq for name, seq in zip(names, gene.sequences)))
            index = {name: i for i, name in enumerate(names)}
        call = index[aln.allele]
        reads, germs = _aligned(aln)
        own = columns[call]
        inside = germs < len(own)
        reads, germs = reads[inside], germs[inside]
        read = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)[reads]
        matches = (bases[:, own[germs]] == read).sum(1)
        tied = matches >= matches[call]
        others = sorted((a for a in np.flatnonzero(tied) if a != call), key=lambda a: (-matches[a], a))
        out.append((names[call], *(names[a] for a in others)))
    return out
