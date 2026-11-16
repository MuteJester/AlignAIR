"""IMGT-gapped alignment construction.

Pure per-read functions over a read's ``GermlineAlignment`` of each gene (``v``, ``d``, ``j``; ``d`` None for a read
without a D placement) and the reference's germlines by allele name (``GermlineMaps``). ``has_d`` says whether the read
is assembled with a D (its locus has one and the D presence head did not gate it out); ``short_d`` that its D is too
short to call, so its germline alignment is the read's own bases.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from GenAIRR.utilities import translate

from ..germline.cigar import ALIGNED, cigar_ops
from ..state import GermlineAlignment


@dataclass(frozen=True)
class GermlineMaps:
    """The germlines the assembly lays a read on, by allele name: the IMGT-gapped V (the ungapped V for a reference
    without gapped forms), and the ungapped J and D (empty for a reference without a D)."""
    v_gapped: Mapping[str, str]
    j: Mapping[str, str]
    d: Mapping[str, str]

    @classmethod
    def of(cls, reference) -> "GermlineMaps":
        """The germline maps of ``reference``."""
        v, j = reference.gene("V"), reference.gene("J")
        d = reference.gene("D") if "D" in reference.genes else None
        return cls(v_gapped=v.gapped or dict(zip(v.names, v.sequences)), j=dict(zip(j.names, j.sequences)),
                   d={} if d is None else dict(zip(d.names, d.sequences)))


def _count_gaps(ref: str) -> int:
    return ref.count(".")


def _germline_placed(seq: str, v: GermlineAlignment) -> list[str] | None:
    """The read base on each V germline position from ``v.germ_start`` to ``v.germ_end`` along ``v.cigar`` (which
    describes ``seq[v.seq_start:v.seq_end]`` against that germline span): ``-`` where the read lacks the germline base,
    and a read base the germline lacks is left out. None when the CIGAR does not span both windows."""
    placed, r = [], v.seq_start
    for count, op in cigar_ops(v.cigar):
        if op in ALIGNED:
            placed.extend(seq[r:r + count])
            r += count
        elif op in "DN":
            placed.extend("-" * count)
        else:                                          # I, S: read bases without a germline position
            r += count
    if not placed or r != v.seq_end or len(placed) != v.germ_end - v.germ_start:
        return None
    return placed


def build_sequence_alignment(seq: str, v_ref_gapped: str, v: GermlineAlignment, j_seq_end: int) -> str | None:
    """The read laid on the IMGT-gapped V germline up to ``v.germ_end``, then the read after V up to ``j_seq_end``.
    Along a ``v.cigar`` that spans the V window each read base sits in its own germline column: a germline base the
    read lacks is ``-`` and a read base the germline lacks is left out (IMGT's indel correction), so a V indel moves
    no later column. Without one the V bases fill the germline columns in order. Germline the read does not cover
    is ``.``."""
    if not v_ref_gapped:
        return None
    v_ref_trimmed = v_ref_gapped[: v.germ_end + _count_gaps(v_ref_gapped)]
    placed = _germline_placed(seq, v)
    seq_to_gap = "".join(placed) if placed is not None else seq[v.seq_start:v.seq_end]
    if v.germ_start > 0:
        seq_to_gap = "." * v.germ_start + seq_to_gap
    seq_iter = iter(seq_to_gap)
    aligned, started = [], False
    for ref_base in v_ref_trimmed:
        if ref_base != ".":
            started = True
            aligned.append(next(seq_iter, "."))
        else:
            aligned.append("." if started else next(seq_iter, "."))
    return "".join(aligned) + seq[v.seq_end:j_seq_end]


def compute_np_regions(seq: str, v: GermlineAlignment, d: GermlineAlignment | None, j: GermlineAlignment,
                       has_d: bool) -> tuple[str, str | None]:
    """The NP regions: with a D, the bases between V and D and between D and J (empty without a D placement);
    without, the bases between V and J, and no second region."""
    if not has_d:
        return seq[v.seq_end:j.seq_start], None
    if d is None:
        return "", ""
    return seq[v.seq_end:d.seq_start], seq[d.seq_end:j.seq_start]


def _germline_along(germ: str, read_len: int, cigar: str) -> str:
    """The germline base on each of a segment's ``read_len`` read bases along ``cigar`` (which describes those read
    bases against ``germ``): ``-`` where the read has a base the germline lacks, and a germline base the read lacks is
    left out, so the segment takes exactly its read's columns. Without a CIGAR spanning both, the germline in order,
    cut to or padded with ``.`` (uncovered) to the read's length."""
    placed, g = [], 0
    for count, op in cigar_ops(cigar):
        if op in ALIGNED:
            placed.append(germ[g:g + count])
            g += count
        elif op == "I":
            placed.append("-" * count)
        elif op == "S":                                # read bases aligned to no germline
            placed.append("." * count)
        else:                                          # D, N: germline bases without a read base
            g += count
    along = "".join(placed)
    if g == len(germ) and len(along) == read_len:
        return along
    return germ[:read_len].ljust(read_len, ".")


def _d_region(seq: str, germlines: GermlineMaps, d_name: str, v: GermlineAlignment, d: GermlineAlignment | None,
              j: GermlineAlignment, np1: str, np2: str, short_d: bool) -> str:
    """The germline between V and J of a read with a D: a short D's is the read's own bases there; otherwise the NP
    bases as they are around the D germline laid on its read bases along its CIGAR."""
    if short_d:
        return seq[v.seq_end:j.seq_start]
    d_ref = germlines.d.get(d_name, "")
    d_part = ""
    if d_ref and d is not None:
        d_part = d_ref[d.germ_start: d.germ_end + _count_gaps(d_ref)]
        if d_part:
            d_part = _germline_along(d_part, max(0, j.seq_start - v.seq_end - len(np1) - len(np2 or "")), d.cigar)
    return np1 + d_part + (np2 or "")


def build_germline_alignment(seq: str, germlines: GermlineMaps, names: Mapping[str, str], v: GermlineAlignment,
                             d: GermlineAlignment | None, j: GermlineAlignment, np1: str, np2: str | None, *,
                             has_d: bool, short_d: bool = False) -> str:
    """The germline under ``build_sequence_alignment``'s columns: the IMGT-gapped V germline up to ``v.germ_end``,
    then for the read after V its NP bases as they are and the D and J germlines each on its own read bases along its
    CIGAR (``_germline_along``), so a D or J indel, or a germline span of another length than its read span, moves no
    later column. ``names`` holds each gene's called allele (the D's empty without a call)."""
    v_ref = germlines.v_gapped.get(names["v"], "")
    v_part = v_ref[:v.germ_end + _count_gaps(v_ref)] if v_ref else ""
    j_ref = germlines.j.get(names["j"], "")
    j_part = j_ref[j.germ_start: j.germ_end + _count_gaps(j_ref)] if j_ref else ""
    if j_part:
        j_part = _germline_along(j_part, j.seq_end - j.seq_start, j.cigar)
    if has_d:
        return v_part + _d_region(seq, germlines, names.get("d", ""), v, d, j, np1, np2, short_d) + j_part
    return v_part + np1 + j_part


def compute_alignment_positions(v_ref_gapped: str, v: GermlineAlignment, d: GermlineAlignment | None,
                                j: GermlineAlignment, has_d: bool) -> dict[str, int | None]:
    """Each segment's positions in the (IMGT-gapped) alignment strings. The V part spans the gapped germline up to
    ``v.germ_end`` (``build_sequence_alignment``) and the read after V follows it base for base, so a read position
    past V sits ``p - v.seq_end`` columns after the V part, wherever V starts in its germline and whatever its indels.
    The germline coordinates stay the aligner's: they are positions in the germline, which the alignment's gaps do
    not move."""
    v_end = v.germ_end + (_count_gaps(v_ref_gapped) if v_ref_gapped else 0)
    d_start_al = d_end_al = None
    if has_d and d is not None:
        d_start_al = v_end + d.seq_start - v.seq_end
        d_end_al = v_end + d.seq_end - v.seq_end
    return {
        "v_alignment_start": 0, "v_alignment_end": v_end,
        "d_alignment_start": d_start_al, "d_alignment_end": d_end_al,
        "j_alignment_start": v_end + j.seq_start - v.seq_end,
        "j_alignment_end": v_end + j.seq_end - v.seq_end,
    }


def extract_segment_alignments(seq_alignment: str, germ_alignment: str, seq_alignment_aa: str,
                               germ_alignment_aa: str, positions: Mapping[str, int | None], has_d: bool) -> dict:
    """Each segment's slice of the read and germline alignments (nucleotide and amino acid) at its ``positions``
    (``compute_alignment_positions``); None for a segment without them."""
    result = {}
    for seg in ["v", "j"] + (["d"] if has_d else []):
        s, e = positions.get(f"{seg}_alignment_start"), positions.get(f"{seg}_alignment_end")
        if s is not None and e is not None:
            result[f"{seg}_sequence_alignment"] = seq_alignment[s:e]
            result[f"{seg}_germline_alignment"] = germ_alignment[s:e]
            result[f"{seg}_sequence_alignment_aa"] = seq_alignment_aa[s // 3:e // 3]
            result[f"{seg}_germline_alignment_aa"] = germ_alignment_aa[s // 3:e // 3]
        else:
            for k in ("sequence_alignment", "germline_alignment",
                      "sequence_alignment_aa", "germline_alignment_aa"):
                result[f"{seg}_{k}"] = None
    return result


def translate_alignment(alignment: str | None) -> str | None:
    """The amino-acid translation of ``alignment`` (GenAIRR's ``translate``), or None for None."""
    return translate(alignment) if alignment is not None else None
