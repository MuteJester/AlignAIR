"""IMGT-gapped alignment construction.

Pure per-sequence functions. Reference germline maps are plain ``{allele_name: seq}`` dicts
(``v_gapped`` = IMGT-gapped V, ``*_ungapped`` = ungapped) sourced from :class:`ReferenceSet`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from GenAIRR.utilities import translate

from ..cigar import ALIGNED, cigar_ops
from .constants import SHORT_D_SENTINEL


def _count_gaps(ref: str) -> int:
    return ref.count(".")


def _reference_segment(ref_seq: str, end: int, start: int = 0) -> str:
    if not ref_seq:
        return ""
    gap_adjusted_end = end + _count_gaps(ref_seq)
    return ref_seq[start:gap_adjusted_end] if start > 0 else ref_seq[:gap_adjusted_end]


def _germline_placed(seq, v_seq_start, v_seq_end, v_germ_start, v_germ_end, v_cigar) -> Optional[List[str]]:
    """The read base on each V germline position from ``v_germ_start`` to ``v_germ_end`` along ``v_cigar`` (which
    describes ``seq[v_seq_start:v_seq_end]`` against that germline span): ``-`` where the read lacks the germline base,
    and a read base the germline lacks is left out. None when the CIGAR does not span both windows."""
    placed, r = [], v_seq_start
    for count, op in cigar_ops(v_cigar):
        if op in ALIGNED:
            placed.extend(seq[r:r + count])
            r += count
        elif op in "DN":
            placed.extend("-" * count)
        else:                                          # I, S: read bases without a germline position
            r += count
    if not placed or r != v_seq_end or len(placed) != v_germ_end - v_germ_start:
        return None
    return placed


def build_sequence_alignment(seq, v_ref_gapped, v_seq_start, v_seq_end, v_germ_start,
                             v_germ_end, j_seq_end, v_cigar=None) -> Optional[str]:
    """The read laid on the IMGT-gapped V germline up to ``v_germ_end``, then the read after V up to ``j_seq_end``.
    Along a ``v_cigar`` that spans the V window each read base sits in its own germline column: a germline base the
    read lacks is ``-`` and a read base the germline lacks is left out (IMGT's indel correction), so a V indel moves
    no later column. Without one the V bases fill the germline columns in order. Germline the read does not cover
    is ``.``."""
    if not v_ref_gapped:
        return None
    gaps = _count_gaps(v_ref_gapped)
    v_ref_trimmed = v_ref_gapped[: v_germ_end + gaps]
    placed = _germline_placed(seq, v_seq_start, v_seq_end, v_germ_start, v_germ_end, v_cigar)
    seq_to_gap = "".join(placed) if placed is not None else seq[v_seq_start:v_seq_end]
    seq_remain = seq[v_seq_end:j_seq_end]
    if v_germ_start > 0:
        seq_to_gap = "." * v_germ_start + seq_to_gap
    seq_iter = iter(seq_to_gap)
    aligned, started = [], False
    for ref_base in v_ref_trimmed:
        if ref_base != ".":
            started = True
            aligned.append(next(seq_iter, "."))
        else:
            aligned.append("." if started else next(seq_iter, "."))
    return "".join(aligned) + seq_remain


def compute_np_regions(seq, v_seq_end, j_seq_start, d_seq_start, d_seq_end,
                       chain) -> Tuple[Optional[str], Optional[str]]:
    if chain == "heavy":
        np1 = seq[v_seq_end:d_seq_start] if d_seq_start is not None else ""
        np2 = seq[d_seq_end:j_seq_start] if d_seq_end is not None else ""
        return np1, np2
    return seq[v_seq_end:j_seq_start], None


def _germline_along(germ, read_len, cigar) -> str:
    """The germline base on each of a segment's ``read_len`` read bases along ``cigar`` (which describes those read
    bases against ``germ``): ``-`` where the read has a base the germline lacks, and a germline base the read lacks is
    left out, so the segment takes exactly its read's columns. Without a CIGAR spanning both, the germline in order,
    cut to or padded with ``.`` (uncovered) to the read's length; without a read length, the germline as it is."""
    if read_len is None:
        return germ
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


def build_germline_alignment(seq, v_gapped, j_ungapped, d_ungapped, v_call, j_call, d_call,
                             v_germ_end, j_germ_start, j_germ_end, d_germ_start, d_germ_end,
                             np1, np2, v_seq_end, j_seq_start, chain, *, j_seq_end=None, d_cigar=None,
                             j_cigar=None) -> Optional[str]:
    """The germline under ``build_sequence_alignment``'s columns: the IMGT-gapped V germline up to ``v_germ_end``, then
    for the read after V its NP bases as they are and the D and J germlines each on its own read bases along its CIGAR
    (``_germline_along``), so a D or J indel, or a germline span of another length than its read span, moves no later
    column. Without ``j_seq_end`` the D and J germline spans are laid end to end as they are."""
    v_ref = v_gapped.get(v_call.split(",")[0], "")
    v_part = _reference_segment(v_ref, v_germ_end)
    j_ref = j_ungapped.get(j_call.split(",")[0], "")
    j_part = j_ref[j_germ_start: j_germ_end + _count_gaps(j_ref)] if j_ref else ""
    if j_part and j_seq_end is not None:
        j_part = _germline_along(j_part, j_seq_end - j_seq_start, j_cigar)
    if chain == "heavy":
        first_d = d_call.split(",")[0] if d_call else ""
        if SHORT_D_SENTINEL in first_d:
            d_region = seq[v_seq_end:j_seq_start]
        else:
            d_ref = d_ungapped.get(first_d, "")
            d_part = d_ref[d_germ_start: d_germ_end + _count_gaps(d_ref)] if d_ref else ""
            if d_part and j_seq_end is not None:
                d_len = max(0, j_seq_start - v_seq_end - len(np1 or "") - len(np2 or ""))
                d_part = _germline_along(d_part, d_len, d_cigar)
            d_region = (np1 or "") + d_part + (np2 or "")
        return v_part + d_region + j_part
    return v_part + (np1 or "") + j_part


def compute_alignment_positions(v_ref_gapped, v_germ_end, v_seq_end, d_seq_start, d_seq_end,
                                j_seq_start, j_seq_end, chain) -> Dict[str, Optional[int]]:
    """Each segment's positions in the (IMGT-gapped) alignment strings. The V part spans the gapped germline up to
    ``v_germ_end`` (``build_sequence_alignment``) and the read after V follows it base for base, so a read position
    past V sits ``p - v_seq_end`` columns after the V part, wherever V starts in its germline and whatever its indels.
    The germline coordinates stay the aligner's: they are positions in the germline, which the alignment's gaps do
    not move."""
    gap = _count_gaps(v_ref_gapped) if v_ref_gapped else 0
    v_end = v_germ_end + gap
    d_start_al = d_end_al = None
    if chain == "heavy" and d_seq_start is not None:
        d_start_al = v_end + d_seq_start - v_seq_end
        d_end_al = v_end + d_seq_end - v_seq_end
    return {
        "v_alignment_start": 0, "v_alignment_end": v_end,
        "d_alignment_start": d_start_al, "d_alignment_end": d_end_al,
        "j_alignment_start": v_end + j_seq_start - v_seq_end,
        "j_alignment_end": v_end + j_seq_end - v_seq_end,
    }


def extract_segment_alignments(seq_alignment, germ_alignment, seq_alignment_aa,
                               germ_alignment_aa, positions, chain) -> dict:
    result = {}
    for seg in ["v", "j"] + (["d"] if chain == "heavy" else []):
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


def translate_alignment(alignment: Optional[str]) -> Optional[str]:
    return translate(alignment) if alignment is not None else None
