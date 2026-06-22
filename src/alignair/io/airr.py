"""Write DNAlignAIR predictions as an AIRR rearrangement TSV (+ calibrated-uncertainty
extension columns the AIRR schema does not cover: per-gene equivalence set, resolution
level, and set confidence)."""
from __future__ import annotations

import csv
from typing import List

GENES = ("v", "d", "j")
# AIRR-standard core columns we populate, then our extensions.
# `sequence` is the CANONICAL (forward-oriented) sequence the coordinates refer to; `rev_comp`
# flags that the input read was reoriented to produce it (so coords always match `sequence`).
# sequence_alignment + the per-gene cigars are AIRR-REQUIRED fields; germline_alignment is
# required-but-emitted-empty (we do not reconstruct the full gapped germline incl. N regions).
_CORE = ["sequence_id", "sequence", "rev_comp", "locus", "v_call", "d_call", "j_call",
         "productive", "junction", "junction_aa", "junction_length",
         "sequence_alignment", "germline_alignment", "v_cigar", "d_cigar", "j_cigar",
         "is_contaminant"]
_COORDS = [f"{g}_{k}" for g in GENES
           for k in ("sequence_start", "sequence_end", "germline_start", "germline_end")]
_EXT = [f"{g}_{k}" for g in GENES for k in ("call_set", "call_level", "set_confidence")]
COLUMNS = _CORE + _EXT + _COORDS


def _airr_start(v):
    """0-based (GenAIRR/our convention) -> 1-based (AIRR)."""
    return (int(v) + 1) if v is not None else None


def _cigar(seq_len, ss, se, gs):
    """An ungapped AIRR CIGAR for one segment from 0-based coords: leading query soft-clip,
    skipped germline prefix (N), the matched span (M), trailing query soft-clip. Correct for the
    substitution-only case (no indels); returns '' if the segment is absent."""
    if ss is None or se is None or se <= ss:
        return ""
    ss, se = int(ss), int(se)
    ops = []
    if ss > 0:
        ops.append(f"{ss}S")
    if gs:
        ops.append(f"{int(gs)}N")
    ops.append(f"{se - ss}M")
    tail = seq_len - se
    if tail > 0:
        ops.append(f"{tail}S")
    return "".join(ops)


def write_airr(path: str, ids: List[str], sequences: List[str], preds: List[dict],
               locus: str = "IGH") -> None:
    """`sequences` must be the CANONICAL (forward-oriented) sequences that predict_reads'
    coordinates are in — use canonicalize_sequence(input, pred['orientation_id']).
    `path` may be '-' to write the TSV to stdout."""
    import sys
    f = sys.stdout if path == "-" else open(path, "w", newline="")
    try:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for sid, seq, p in zip(ids, sequences, preds):
            isc = p.get("is_contaminant")
            seq_len = len(seq)
            # aligned span of the query (V start .. J end); AIRR sequence_alignment
            starts = [p.get(f"{g}_sequence_start") for g in GENES if p.get(f"{g}_sequence_start") is not None]
            ends = [p.get(f"{g}_sequence_end") for g in GENES if p.get(f"{g}_sequence_end")]
            seq_aln = seq[min(starts):max(ends)] if starts and ends else ""
            row = {"sequence_id": sid, "sequence": seq, "locus": locus,
                   "rev_comp": "T" if p.get("orientation_id", 0) != 0 else "F",
                   "productive": p.get("productive"),
                   "junction": p.get("junction"), "junction_aa": p.get("junction_aa"),
                   "junction_length": p.get("junction_length"),
                   "sequence_alignment": seq_aln, "germline_alignment": "",
                   "v_cigar": _cigar(seq_len, p.get("v_sequence_start"), p.get("v_sequence_end"), p.get("v_germline_start")),
                   "d_cigar": _cigar(seq_len, p.get("d_sequence_start"), p.get("d_sequence_end"), p.get("d_germline_start")),
                   "j_cigar": _cigar(seq_len, p.get("j_sequence_start"), p.get("j_sequence_end"), p.get("j_germline_start")),
                   "is_contaminant": ("T" if isc else "F") if isc is not None else None}
            for g in GENES:
                row[f"{g}_call"] = p.get(f"{g}_call")
                row[f"{g}_sequence_start"] = _airr_start(p.get(f"{g}_sequence_start"))
                row[f"{g}_sequence_end"] = p.get(f"{g}_sequence_end")
                row[f"{g}_germline_start"] = _airr_start(p.get(f"{g}_germline_start"))
                row[f"{g}_germline_end"] = p.get(f"{g}_germline_end")
                cset = p.get(f"{g}_call_set") or ([p.get(f"{g}_call")] if p.get(f"{g}_call") else [])
                row[f"{g}_call_set"] = ",".join(c for c in cset if c)
                row[f"{g}_call_level"] = p.get(f"{g}_call_level")
                conf = p.get(f"{g}_set_confidence")
                row[f"{g}_set_confidence"] = f"{conf:.4f}" if conf is not None else None
            w.writerow(row)
    finally:
        if path != "-":
            f.close()
