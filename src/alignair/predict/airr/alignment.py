"""IMGT-gapped alignment construction.

Pure per-sequence functions. Reference germline maps are plain ``{allele_name: seq}`` dicts
(``v_gapped`` = IMGT-gapped V, ``*_ungapped`` = ungapped) sourced from :class:`ReferenceSet`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from GenAIRR.utilities import translate

from .constants import SHORT_D_SENTINEL


def _count_gaps(ref: str) -> int:
    return ref.count(".")


def _reference_segment(ref_seq: str, end: int, start: int = 0) -> str:
    if not ref_seq:
        return ""
    gap_adjusted_end = end + _count_gaps(ref_seq)
    return ref_seq[start:gap_adjusted_end] if start > 0 else ref_seq[:gap_adjusted_end]


def build_sequence_alignment(seq, v_ref_gapped, v_seq_start, v_seq_end, v_germ_start,
                             v_germ_end, j_seq_end) -> Optional[str]:
    """Insert IMGT gaps into the query to match the V-germline gapped reference, then append D+J."""
    if not v_ref_gapped:
        return None
    gaps = _count_gaps(v_ref_gapped)
    v_ref_trimmed = v_ref_gapped[: v_germ_end + gaps]
    seq_to_gap = seq[v_seq_start:v_seq_end]
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


def build_germline_alignment(seq, v_gapped, j_ungapped, d_ungapped, v_call, j_call, d_call,
                             v_germ_end, j_germ_start, j_germ_end, d_germ_start, d_germ_end,
                             np1, np2, v_seq_end, j_seq_start, chain) -> Optional[str]:
    v_ref = v_gapped.get(v_call.split(",")[0], "")
    v_part = _reference_segment(v_ref, v_germ_end)
    j_ref = j_ungapped.get(j_call.split(",")[0], "")
    j_part = j_ref[j_germ_start: j_germ_end + _count_gaps(j_ref)] if j_ref else ""
    if chain == "heavy":
        first_d = d_call.split(",")[0] if d_call else ""
        if SHORT_D_SENTINEL in first_d:
            d_region = seq[v_seq_end:j_seq_start]
        else:
            d_ref = d_ungapped.get(first_d, "")
            d_part = d_ref[d_germ_start: d_germ_end + _count_gaps(d_ref)] if d_ref else ""
            d_region = (np1 or "") + d_part + (np2 or "")
        return v_part + d_region + j_part
    return v_part + (np1 or "") + j_part


def compute_alignment_positions(v_ref_gapped, v_germ_end, v_seq_start, d_seq_start, d_seq_end,
                                j_seq_start, j_seq_end, chain) -> Dict[str, Optional[int]]:
    gap = _count_gaps(v_ref_gapped) if v_ref_gapped else 0
    v_end = v_germ_end + gap
    d_start_al = d_end_al = None
    if chain == "heavy" and d_seq_start is not None:
        d_start_al = d_seq_start - v_seq_start + gap
        d_end_al = d_seq_end - v_seq_start + gap
    return {
        "v_alignment_start": 0, "v_alignment_end": v_end,
        "v_germline_start": 0, "v_germline_end": v_end,
        "d_alignment_start": d_start_al, "d_alignment_end": d_end_al,
        "j_alignment_start": j_seq_start - v_seq_start + gap,
        "j_alignment_end": j_seq_end - v_seq_start + gap,
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
