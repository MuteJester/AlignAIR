"""Germline alignment: place each read's top-1 allele segment in its germline, producing final
read/germline coordinates + a CIGAR.

The anchored :class:`HeuristicGermlineMatcher` locates the segment within the germline (robust to mutated/5'- and
3'-truncated ends, no drift), and the pinned-window banded DP (:func:`derive_alignment`, C-speed via the compiled
``_derive_cy`` kernel, pure-Python fallback) reconstructs the M/I/D CIGAR. Needs no native aligner dependency.

``Short-D`` (empty germline) is a sentinel: a zero-length germline window meaning "no real D".
"""
from __future__ import annotations

from .state import GermlineAlignment


def _germline_seq(gene_ref, allele: str) -> str:
    idx = gene_ref.index.get(allele)
    return gene_ref.sequences[idx].upper() if idx is not None else ""


def _heuristic_alignment(seq, s, e, allele, germ, indel, matcher, derive) -> GermlineAlignment:
    gm = matcher.match_one(seq, s, e, allele, indel)        # robust anchored localization (coords)
    cigar = derive(seq[gm.seq_start:gm.seq_end], germ[gm.ref_start:gm.ref_end], indel).cigar
    return GermlineAlignment(allele, gm.seq_start, gm.seq_end, gm.ref_start, gm.ref_end, cigar)


def _gapless(seq, s, e, aln: GermlineAlignment, germ) -> GermlineAlignment:
    """A D aligned without gaps, as IgBLAST and IMGT align one: ``aln`` itself when it has none, else the read's D
    bases ``seq[s:e]`` at the offset along ``germ`` that matches most of them (among equals the one nearest
    ``aln``'s placement of the first base), cut to the bases the germline covers; a D of no read bases spans no germline
    base."""
    if "I" not in aln.cigar and "D" not in aln.cigar:
        return aln
    segment, n, m = seq[s:e].upper(), e - s, len(germ)
    if n <= 0:
        return GermlineAlignment(aln.allele, s, s, 0, 0, "")
    nearest = aln.germ_start - (aln.seq_start - s)
    best = None
    for offset in range(1 - n, m):
        lo, hi = max(0, -offset), min(n, m - offset)
        matches = sum(a == b for a, b in zip(segment[lo:hi], germ[lo + offset:hi + offset]))
        key = (matches, -abs(offset - nearest))
        if best is None or key > best[0]:
            best = (key, offset, lo, hi)
    _, offset, lo, hi = best
    return GermlineAlignment(aln.allele, s + lo, s + hi, lo + offset, hi + offset, f"{hi - lo}M")


def align_germline(sequences, segments, calls, reference, *, indel_counts=None) -> dict:
    """Returns {gene: list[GermlineAlignment | None]} (None when a read has no call for the gene).

    ``indel_counts`` (per-read, e.g. the model's predicted indel count) bounds the heuristic reader's
    search window and CIGAR band; defaults to 0 per read when absent.
    """
    from .heuristic_matcher import HeuristicGermlineMatcher, derive_alignment as derive
    genes = list(segments.start)
    germlines: dict[str, str] = {}
    for gene in genes:
        gref = reference.gene(gene.upper())
        germlines.update({n: s.upper() for n, s in zip(gref.names, gref.sequences)})
    matcher = HeuristicGermlineMatcher(germlines)

    out: dict[str, list] = {}
    for gene in genes:
        gene_ref = reference.gene(gene.upper())
        seg_s, seg_e = segments.start[gene], segments.end[gene]
        results = []
        for i, seq in enumerate(sequences):
            call = calls[gene][i]
            if not call.names:
                results.append(None)
                continue
            allele = call.names[0]
            s, e = int(seg_s[i]), int(seg_e[i])
            germ = _germline_seq(gene_ref, allele)
            if not germ:                                    # Short-D / empty-reference sentinel
                results.append(GermlineAlignment(allele, s, s, 0, 0, ""))
                continue
            ind = float(indel_counts[i]) if indel_counts is not None else 0.0
            aln = _heuristic_alignment(seq, s, e, allele, germ, ind, matcher, derive)
            results.append(_gapless(seq, s, e, aln, germ) if gene == "d" else aln)
        out[gene] = results
    return out
