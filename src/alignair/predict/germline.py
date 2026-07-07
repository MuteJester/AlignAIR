"""Germline alignment: align each read's top-1 allele segment to its germline, producing final
read/germline coordinates + a CIGAR.

Deviation from TF (deliberate, faithful-to-intent): TF used a hand-rolled affine offset search that
produced only 4 integer coords and no CIGAR. We reuse ``alignair.align`` (WFA/parasail, banded) — more
principled, and it yields a real CIGAR. Only the top-1 call is aligned (as in TF). ``Short-D`` (empty
germline) is a sentinel: a zero-length germline window meaning "no real D".
"""
from __future__ import annotations

import numpy as np

from ..align import get_aligner
from .state import GermlineAlignment


def _germline_seq(gene_ref, allele: str) -> str:
    idx = gene_ref.index.get(allele)
    return gene_ref.sequences[idx].upper() if idx is not None else ""


def align_germline(sequences, segments, calls, reference, aligner=None) -> dict:
    """Returns {gene: list[GermlineAlignment | None]} (None when a read has no call for the gene)."""
    aligner = aligner or get_aligner()
    out: dict[str, list] = {}
    for gene in segments.start:
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
            r = aligner.align(seq[s:e], germ)
            if r is None:                                   # no alignment -> trust the segment span
                results.append(GermlineAlignment(allele, s, e, 0, len(germ), f"{e - s}M"))
                continue
            results.append(GermlineAlignment(allele, s + r.q_start, s + r.q_end,
                                             r.t_start, r.t_end, r.cigar))
        out[gene] = results
    return out
