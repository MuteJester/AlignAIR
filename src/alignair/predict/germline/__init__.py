"""Germline alignment: each read's top allele segment placed in its germline, as read and germline coordinates and a
CIGAR. ``locate`` places the segment (robust to mutated and 5'- or 3'-truncated ends), ``cigar.derive_cigar`` derives
its CIGAR on the located window (a pinned banded DP, in the compiled ``_derive_cy`` kernel when it is built) and
``settle`` moves a V alignment's ends where it fits better."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..state import GeneCall, GermlineAlignment, Segments
from .cigar import derive_cigar, matched_bases
from .locate import GermlineLocator
from .settle import settle_end, settle_start

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...reference.reference_set import ReferenceSet


def _germline_seq(gene_ref, allele: str) -> str:
    idx = gene_ref.index.get(allele)
    return gene_ref.sequences[idx].upper() if idx is not None else ""


def _aligned(seq, s, e, allele, germ, indel, locator, *, limit: int | None = None) -> GermlineAlignment:
    """The locator's placement of ``seq[s:e]`` on ``germ`` and its CIGAR. With a ``limit`` (V: the read position the
    next segment starts at), the germline start is then re-placed where the alignment fits best (``settle_start``),
    and the end moved where it does (``settle_end``), the read end no further than ``limit``."""
    gm = locator.locate(seq, s, e, allele, indel)
    seq_end, ref_start, ref_end = gm.seq_end, gm.ref_start, gm.ref_end
    cigar = derive_cigar(seq[gm.seq_start:seq_end], germ[ref_start:ref_end], indel)
    if limit is not None:
        upper = seq.upper()
        ref_start, cigar = settle_start(upper[gm.seq_start:seq_end], germ, ref_start, ref_end, indel, cigar)
        end = settle_end(upper, germ, gm.seq_start, seq_end, ref_start, ref_end, cigar, limit)
        if end != (seq_end, ref_end):
            seq_end, ref_end = end
            cigar = derive_cigar(upper[gm.seq_start:seq_end], germ[ref_start:ref_end], indel)
    return GermlineAlignment(allele, gm.seq_start, seq_end, ref_start, ref_end, cigar)


def v_end_limits(segments, lengths) -> list[int]:
    """Per read, the read position a V alignment may reach: the start of the nearest later segment of the read's
    other genes (``segments``, any genes but V), else the read's length (``lengths``)."""
    others = [g for g in segments.start if g != "v"]
    v_start = segments.start.get("v")
    out = []
    for i, length in enumerate(lengths):
        s = int(v_start[i]) if v_start is not None else 0
        later = [int(segments.start[g][i]) for g in others if int(segments.start[g][i]) > s]
        out.append(min(later, default=int(length)))
    return out


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


def align_germline(sequences: Sequence[str], segments: Segments, calls: dict[str, list[GeneCall]],
                   reference: ReferenceSet, *, indel_counts=None,
                   v_limits=None) -> dict[str, list[GermlineAlignment | None]]:
    """Returns {gene: list[GermlineAlignment | None]} (None when a read has no call for the gene).

    ``indel_counts`` (per-read, e.g. the model's predicted indel count) bounds the locator's search window and the
    CIGAR band; defaults to 0 per read when absent. ``v_limits`` (per read) bounds where a V end may
    move to; defaults to ``v_end_limits`` of ``segments``.
    """
    genes = list(segments.start)
    germlines: dict[str, str] = {}
    for gene in genes:
        gref = reference.gene(gene.upper())
        germlines.update({n: s.upper() for n, s in zip(gref.names, gref.sequences)})
    locator = GermlineLocator(germlines)
    if v_limits is None and "v" in segments.start:
        v_limits = v_end_limits(segments, [len(seq) for seq in sequences])

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
            if not germ:                                    # an allele with no germline sequence: an empty alignment
                results.append(GermlineAlignment(allele, s, s, 0, 0, ""))
                continue
            ind = float(indel_counts[i]) if indel_counts is not None else 0.0
            aln = _aligned(seq, s, e, allele, germ, ind, locator, limit=v_limits[i] if gene == "v" else None)
            results.append(_gapless(seq, s, e, aln, germ) if gene == "d" else aln)
        out[gene] = results
    return out


def _matched(sequence: str, aln: GermlineAlignment, germ: str) -> int:
    """How many read bases ``aln`` puts on an equal germline base (its CIGAR's M columns)."""
    return matched_bases(sequence.upper(), germ, aln.cigar, aln.seq_start, aln.germ_start)


def realign_v(sequences: Sequence[str], segments: Segments, alignments: list[GermlineAlignment | None],
              alleles: Sequence[str | None], reference: ReferenceSet, *,
              indel_counts=None) -> list[GermlineAlignment | None]:
    """``alignments`` (the reads' V alignments) with read i's replaced by the alignment of ``alleles[i]``, aligned as
    ``align_germline`` aligns a called allele over the read's V segment (``segments``), where ``alleles[i]`` is not
    None and that alignment puts more read bases on equal germline bases than read i's current one; the other reads
    keep theirs. ``predict.ties.better_v_alleles`` names the alleles by comparing them at IMGT columns, which charges
    an allele nothing for germline bases the read lacks, so the realignment decides."""
    redo = [i for i, allele in enumerate(alleles) if allele is not None]
    if not redo:
        return list(alignments)
    picked = Segments({"v": segments.start["v"][redo]}, {"v": segments.end["v"][redo]})
    calls = {"v": [GeneCall((alleles[i],), (float("nan"),)) for i in redo]}
    counts = None if indel_counts is None else [indel_counts[i] for i in redo]
    limits = v_end_limits(segments, [len(seq) for seq in sequences])
    new = align_germline([sequences[i] for i in redo], picked, calls, reference, indel_counts=counts,
                         v_limits=[limits[i] for i in redo])["v"]
    gene = reference.gene("V")
    out = list(alignments)
    for i, aln in zip(redo, new, strict=True):
        current = alignments[i]
        if current is None or _matched(sequences[i], aln, _germline_seq(gene, aln.allele)) > _matched(
                sequences[i], current, _germline_seq(gene, current.allele)):
            out[i] = aln
    return out
