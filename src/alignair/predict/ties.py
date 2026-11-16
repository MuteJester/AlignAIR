"""The V alleles a read cannot tell from its call: the record's ``v_call`` lists them after the call, as IgBLAST lists
equally scoring alleles, so a genotype step downstream (TIgGER, Change-O) can choose among them.

An allele ties with the call when it matches at least as many of the read's V bases as the call does: the read bases
the call's alignment puts on germline bases (its CIGAR's M columns), each compared with the allele's base at the same
IMGT column (the IMGT gapped forms put alleles of other lengths and with internal gaps in one frame; a reference
without them compares germline positions). An allele without a base at a column (a gap, or past its end) matches no
read base there, and an inserted read base is compared with no allele. Only an allele of the bound reference ties (a
genotype is the bank, so its alleles are the only candidates). The ties follow the call by matches, then in reference
order.

An allele matching strictly more of those read bases than the call is better than it (``better_v_alleles``): the read
shows a base only that allele holds, so the prediction makes it the call (and realigns V to it) before the ties are
listed. This is what lets a bound reference's alleles the model never trained on, a novel allele among them, be called.
"""
from __future__ import annotations

import functools

import numpy as np

from .germline.cigar import aligned_columns

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


class _Bank:
    """The bound reference's V alleles as the comparison reads them: names, their index, base codes and columns."""

    def __init__(self, reference):
        gene = reference.gene("V")
        self.names, forms = list(gene.names), gene.gapped or {}
        self.bases, self.columns = _table(tuple(forms.get(name) or seq
                                                for name, seq in zip(self.names, gene.sequences)))
        self.index = {name: i for i, name in enumerate(self.names)}

    def matches(self, sequence, aln) -> tuple:
        """``(matches, call)``: per allele, how many of the read bases the call's alignment ``aln`` puts on germline
        bases it matches at their IMGT columns, and the call's row."""
        call = self.index[aln.allele]
        reads, germs = aligned_columns(aln.cigar, aln.seq_start, aln.germ_start)
        own = self.columns[call]
        inside = germs < len(own)
        reads, germs = reads[inside], germs[inside]
        read = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)[reads]
        return (self.bases[:, own[germs]] == read).sum(1), call


def _allowed(matches, allowed, i):
    """``matches`` with the alleles outside read i's ``allowed`` (a bool array over the V alleles; None: all) at -1."""
    if allowed is None or allowed[i] is None:
        return matches
    return np.where(allowed[i], matches, -1)


def better_v_alleles(sequences, alignments, reference, candidates, allowed=None) -> list:
    """Per read (forward frame), the V allele of the bound bank that matches strictly more of the read bases its V
    alignment puts on germline bases than the call does, or None (no alignment, or the call matches most). Among the
    alleles matching most, the first of the read's ``candidates`` (the model's called names, in its order) that is one,
    else the first in reference order. ``allowed`` (per read, a bool array over the V alleles, or None) keeps a read
    to its locus's alleles (``refine.locus``): no other allele is better."""
    bank = None
    out = []
    for i, (sequence, aln, named) in enumerate(zip(sequences, alignments, candidates, strict=True)):
        if aln is None:
            out.append(None)
            continue
        bank = bank or _Bank(reference)
        matches, call = bank.matches(sequence, aln)
        matches = _allowed(matches, allowed, i)
        top = matches.max()
        if matches[call] >= top:
            out.append(None)
            continue
        best = set(np.flatnonzero(matches == top).tolist())
        pick = next((bank.index[n] for n in named if bank.index.get(n) in best), min(best))
        out.append(bank.names[pick])
    return out


def tied_alleles(sequences, alignments, reference, allowed=None) -> list:
    """Per read (forward frame), its V call and the V alleles of the bound bank tied with it, the call first (a tuple
    of names), or ``()`` for a read without a V alignment. ``allowed`` (per read, a bool array over the V alleles, or
    None) keeps a read's ties to its locus's alleles (``refine.locus``)."""
    bank = None
    out = []
    for i, (sequence, aln) in enumerate(zip(sequences, alignments, strict=True)):
        if aln is None:
            out.append(())
            continue
        bank = bank or _Bank(reference)
        names = bank.names
        matches, call = bank.matches(sequence, aln)
        tied = (matches >= matches[call]) & (_allowed(matches, allowed, i) >= 0)
        others = sorted((a for a in np.flatnonzero(tied) if a != call), key=lambda a: (-matches[a], a))
        out.append((names[call], *(names[a] for a in others)))
    return out
