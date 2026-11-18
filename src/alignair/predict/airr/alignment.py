"""IMGT-gapped alignment construction.

Pure per-read functions over a read's ``GermlineAlignment`` of each gene (``v``, ``d``, ``j``; ``d`` None for a read
without a D placement) and the reference's germlines by allele name (``GermlineMaps``). ``has_d`` says whether the read
is assembled with a D region (its locus has one and the D presence head did not find it absent); ``d_called`` that the
D is also called (not too short to call), so the bases around it are its NP regions and its germline lies under it.
Positions are 0-based and end-exclusive, in the read or in the alignment strings; the AIRR writer
(``io.airr``) writes them 1-based and closed.
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
    def of(cls, reference) -> GermlineMaps:
        """The germline maps of ``reference``."""
        v, j = reference.gene("V"), reference.gene("J")
        d = reference.gene("D") if "D" in reference.genes else None
        return cls(v_gapped=v.gapped or dict(zip(v.names, v.sequences)), j=dict(zip(j.names, j.sequences)),
                   d={} if d is None else dict(zip(d.names, d.sequences)))


def _count_gaps(ref: str) -> int:
    return ref.count(".")


def _germline_placed(seq: str, v: GermlineAlignment) -> list[tuple[str, int | None]] | None:
    """The read base on each V germline position from ``v.germ_start`` to ``v.germ_end`` along ``v.cigar`` (which
    describes ``seq[v.seq_start:v.seq_end]`` against that germline span), with its read position: ``-`` (no position)
    where the read lacks the germline base, and a read base the germline lacks is left out. None when the CIGAR does
    not span both windows."""
    placed, r = [], v.seq_start
    for count, op in cigar_ops(v.cigar):
        if op in ALIGNED:
            placed.extend((seq[p], p) for p in range(r, min(r + count, len(seq))))
            r += count
        elif op in "DN":
            placed.extend([("-", None)] * count)
        else:                                          # I, S: read bases without a germline position
            r += count
    if not placed or r != v.seq_end or len(placed) != v.germ_end - v.germ_start:
        return None
    return placed


def _v_columns(seq: str, v_ref_gapped: str, v: GermlineAlignment) -> list[tuple[str, int | None]]:
    """Each column of the V part of the read's alignment, the IMGT-gapped V germline up to ``v.germ_end``: its
    character and the read position of its base (None for a ``.`` or ``-``). Along a ``v.cigar`` that spans the V
    window each read base sits in its own germline column; without one the V bases fill the germline columns in
    order. Germline the read does not cover is ``.``."""
    v_ref_trimmed = v_ref_gapped[: v.germ_end + _count_gaps(v_ref_gapped)]
    placed = _germline_placed(seq, v)
    if placed is None:
        placed = [(seq[p], p) for p in range(v.seq_start, min(v.seq_end, len(seq)))]
    cells = iter([(".", None)] * v.germ_start + placed)
    columns, started = [], False
    for ref_base in v_ref_trimmed:
        if ref_base != ".":
            started = True
            columns.append(next(cells, (".", None)))
        else:
            columns.append((".", None) if started else next(cells, (".", None)))
    return columns


def sequence_alignment_columns(seq: str, v_ref_gapped: str, v: GermlineAlignment,
                               j_seq_end: int) -> tuple[str, list[int | None]] | None:
    """``build_sequence_alignment``'s alignment and, for each of its columns, the read position of the base in it
    (None for a gap column, a base the read lacks, or germline the read does not cover); None without a V germline.
    A region of the alignment spans the read from its first to its last position."""
    if not v_ref_gapped:
        return None
    columns = _v_columns(seq, v_ref_gapped, v)
    after = range(v.seq_end, min(j_seq_end, len(seq)))
    return ("".join(c for c, _ in columns) + seq[v.seq_end:j_seq_end],
            [p for _, p in columns] + list(after))


def build_sequence_alignment(seq: str, v_ref_gapped: str, v: GermlineAlignment, j_seq_end: int) -> str | None:
    """The read laid on the IMGT-gapped V germline up to ``v.germ_end``, then the read after V up to ``j_seq_end``.
    Along a ``v.cigar`` that spans the V window each read base sits in its own germline column: a germline base the
    read lacks is ``-`` and a read base the germline lacks is left out (IMGT's indel correction), so a V indel moves
    no later column. Without one the V bases fill the germline columns in order. Germline the read does not cover
    is ``.``."""
    built = sequence_alignment_columns(seq, v_ref_gapped, v, j_seq_end)
    return None if built is None else built[0]


def compute_np_regions(seq: str, v: GermlineAlignment, d: GermlineAlignment | None, j: GermlineAlignment,
                       d_called: bool) -> tuple[str, str | None]:
    """The NP regions: with a called D, the bases between V and D and between D and J; without one (a locus without
    a D, a D the presence head found absent or too short to call), every base between V and J, and no second region,
    as Change-O reads a row without a D call."""
    if not d_called or d is None:
        return seq[v.seq_end:j.seq_start], None
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


def build_germline_alignment(germlines: GermlineMaps, names: Mapping[str, str], v: GermlineAlignment,
                             d: GermlineAlignment | None, j: GermlineAlignment, np1: str, np2: str | None, *,
                             d_called: bool) -> str:
    """The germline under ``build_sequence_alignment``'s columns: the IMGT-gapped V germline up to ``v.germ_end``,
    then N for each untemplated base (``np1``, ``np2``) and the called D and the J germlines each on its own read bases
    along its CIGAR (``_germline_along``), so a D or J indel, or a germline span of another length than its read span,
    moves no later column. ``names`` holds each gene's called allele; without a called D (``d_called``) every base
    between V and J is untemplated (``compute_np_regions``)."""
    v_ref = germlines.v_gapped.get(names["v"], "")
    v_part = v_ref[:v.germ_end + _count_gaps(v_ref)] if v_ref else ""
    j_ref = germlines.j.get(names["j"], "")
    j_part = j_ref[j.germ_start: j.germ_end + _count_gaps(j_ref)] if j_ref else ""
    if j_part:
        j_part = _germline_along(j_part, j.seq_end - j.seq_start, j.cigar)
    middle = "N" * len(np1)
    if d_called and d is not None:
        d_ref = germlines.d.get(names.get("d", ""), "")
        d_part = d_ref[d.germ_start: d.germ_end + _count_gaps(d_ref)] if d_ref else ""
        if d_part:
            d_part = _germline_along(d_part, max(0, j.seq_start - v.seq_end - len(np1) - len(np2 or "")), d.cigar)
        middle += d_part + "N" * len(np2 or "")
    return v_part + middle + j_part


def compute_alignment_positions(v_ref_gapped: str, v: GermlineAlignment, d: GermlineAlignment | None,
                                j: GermlineAlignment, has_d: bool) -> dict[str, int | None]:
    """Each segment's positions in the (IMGT-gapped) alignment strings. The V part spans the gapped germline up to
    ``v.germ_end`` (``build_sequence_alignment``) and the read after V follows it base for base, so a read position
    past V sits ``p - v.seq_end`` columns after the V part, wherever V starts in its germline and whatever its indels.
    The V part starts at the gapped germline's first position, so its alignment positions are also its positions in
    the IMGT-gapped V germline (the AIRR table's ``v_germline_start``/``_end``, as Change-O counts them). The record's
    germline coordinates stay the aligner's: positions in the ungapped germline, which the alignment's gaps do not
    move."""
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
