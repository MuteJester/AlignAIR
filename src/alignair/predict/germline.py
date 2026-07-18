"""Germline alignment: place each read's top-1 allele segment in its germline, producing final
read/germline coordinates + a CIGAR.

Two readers (selectable via ``reader``):

  * ``"heuristic"`` (default) — the anchored :class:`HeuristicGermlineMatcher` locates the segment
    within the germline (robust to mutated/5'- and 3'-truncated ends, no drift), and the pinned-window
    banded DP (:func:`derive_alignment`, C-speed via the compiled ``_derive_cy`` kernel, pure-Python
    fallback) reconstructs the M/I/D CIGAR. Needs no native aligner dependency.
  * ``"wfa"`` — the ends-free WFA/parasail aligner (:mod:`alignair.align`); kept for comparison.

``Short-D`` (empty germline) is a sentinel: a zero-length germline window meaning "no real D".
"""
from __future__ import annotations

from .state import GermlineAlignment


def _germline_seq(gene_ref, allele: str) -> str:
    idx = gene_ref.index.get(allele)
    return gene_ref.sequences[idx].upper() if idx is not None else ""


def _wfa_alignment(seq, s, e, allele, germ, aligner) -> GermlineAlignment:
    r = aligner.align(seq[s:e], germ)
    if r is None:                                           # no alignment -> trust the segment span
        return GermlineAlignment(allele, s, e, 0, len(germ), f"{e - s}M")
    return GermlineAlignment(allele, s + r.q_start, s + r.q_end, r.t_start, r.t_end, r.cigar)


def _heuristic_alignment(seq, s, e, allele, germ, indel, matcher, derive) -> GermlineAlignment:
    gm = matcher.match_one(seq, s, e, allele, indel)        # robust anchored localization (coords)
    cigar = derive(seq[gm.seq_start:gm.seq_end], germ[gm.ref_start:gm.ref_end], indel).cigar
    return GermlineAlignment(allele, gm.seq_start, gm.seq_end, gm.ref_start, gm.ref_end, cigar)


def align_germline(sequences, segments, calls, reference, aligner=None, *,
                   reader: str = "heuristic", indel_counts=None) -> dict:
    """Returns {gene: list[GermlineAlignment | None]} (None when a read has no call for the gene).

    ``indel_counts`` (per-read, e.g. the model's predicted indel count) bounds the heuristic reader's
    search window and CIGAR band; defaults to 0 per read when absent.
    """
    genes = list(segments.start)
    matcher = derive = None
    if reader == "heuristic":
        from .heuristic_matcher import HeuristicGermlineMatcher, derive_alignment as derive
        germlines: dict[str, str] = {}
        for gene in genes:
            gref = reference.gene(gene.upper())
            germlines.update({n: s.upper() for n, s in zip(gref.names, gref.sequences)})
        matcher = HeuristicGermlineMatcher(germlines)
    else:
        from ..align import get_aligner
        aligner = aligner or get_aligner()

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
            if reader == "heuristic":
                ind = float(indel_counts[i]) if indel_counts is not None else 0.0
                results.append(_heuristic_alignment(seq, s, e, allele, germ, ind, matcher, derive))
            else:
                results.append(_wfa_alignment(seq, s, e, allele, germ, aligner))
        out[gene] = results
    return out
