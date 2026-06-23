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
         "productive", "vj_in_frame", "stop_codon",
         "junction", "junction_aa", "junction_length", "np1", "np1_length", "np2", "np2_length",
         "sequence_alignment", "germline_alignment", "v_cigar", "d_cigar", "j_cigar",
         "v_identity", "d_identity", "j_identity", "is_contaminant"]
_COORDS = [f"{g}_{k}" for g in GENES
           for k in ("sequence_start", "sequence_end", "germline_start", "germline_end")]
_EXT = [f"{g}_{k}" for g in GENES for k in ("call_set", "resolved_call", "call_level", "set_confidence")]
COLUMNS = _CORE + _EXT + _COORDS


def _airr_start(v):
    """0-based (GenAIRR/our convention) -> 1-based (AIRR)."""
    return (int(v) + 1) if v is not None else None


def _cigar(seq_len, ss, se, gs, ge):
    """An AIRR CIGAR for one segment from 0-based coords: leading query soft-clip (S), skipped
    germline prefix (N), the matched span (M), a single net indel op when the read and germline
    spans differ (I if the query is longer, D if the germline is longer), then trailing query
    soft-clip. This is a coordinate-derived approximation (the net indel is placed at the segment
    end, not its true position) but is length-consistent with both the query and germline spans;
    returns '' if the segment is absent."""
    if ss is None or se is None or se <= ss:
        return ""
    ss, se = int(ss), int(se)
    read_span = se - ss
    germ_span = (int(ge) - int(gs)) if (gs is not None and ge is not None) else read_span
    m = min(read_span, germ_span)
    ops = []
    if ss > 0:
        ops.append(f"{ss}S")
    if gs:
        ops.append(f"{int(gs)}N")
    ops.append(f"{m}M")
    if read_span > m:
        ops.append(f"{read_span - m}I")
    elif germ_span > m:
        ops.append(f"{germ_span - m}D")
    tail = seq_len - se
    if tail > 0:
        ops.append(f"{tail}S")
    return "".join(ops)


def _build_row(sid: str, seq: str, p: dict, locus: str) -> dict:
    """Build one AIRR rearrangement row. `seq` must be the CANONICAL (forward) sequence the
    coordinates are in. Prefers real-alignment fields (predict_reads full_alignment) when present,
    else the coordinate approximation."""
    isc = p.get("is_contaminant")
    seq_len = len(seq)
    starts = [p.get(f"{g}_sequence_start") for g in GENES if p.get(f"{g}_sequence_start") is not None]
    ends = [p.get(f"{g}_sequence_end") for g in GENES if p.get(f"{g}_sequence_end")]
    seq_aln = p.get("sequence_alignment")
    if seq_aln is None:
        seq_aln = seq[min(starts):max(ends)] if starts and ends else ""
    row = {"sequence_id": sid, "sequence": seq, "locus": locus,
           "rev_comp": "T" if p.get("orientation_id", 0) != 0 else "F",
           "productive": p.get("productive"),
           "vj_in_frame": p.get("vj_in_frame"), "stop_codon": p.get("stop_codon"),
           "junction": p.get("junction"), "junction_aa": p.get("junction_aa"),
           "junction_length": p.get("junction_length"),
           "np1": p.get("np1"), "np1_length": p.get("np1_length"),
           "np2": p.get("np2"), "np2_length": p.get("np2_length"),
           "sequence_alignment": seq_aln, "germline_alignment": p.get("germline_alignment", ""),
           "v_identity": p.get("v_identity"), "d_identity": p.get("d_identity"),
           "j_identity": p.get("j_identity"),
           "is_contaminant": ("T" if isc else "F") if isc is not None else None}
    for g in GENES:
        row[f"{g}_cigar"] = p.get(f"{g}_cigar") or _cigar(
            seq_len, p.get(f"{g}_sequence_start"), p.get(f"{g}_sequence_end"),
            p.get(f"{g}_germline_start"), p.get(f"{g}_germline_end"))
        row[f"{g}_call"] = p.get(f"{g}_call")
        row[f"{g}_sequence_start"] = _airr_start(p.get(f"{g}_sequence_start"))
        row[f"{g}_sequence_end"] = p.get(f"{g}_sequence_end")
        row[f"{g}_germline_start"] = _airr_start(p.get(f"{g}_germline_start"))
        row[f"{g}_germline_end"] = p.get(f"{g}_germline_end")
        cset = p.get(f"{g}_call_set") or ([p.get(f"{g}_call")] if p.get(f"{g}_call") else [])
        row[f"{g}_call_set"] = ",".join(c for c in cset if c)
        row[f"{g}_resolved_call"] = p.get(f"{g}_resolved_call")    # most-specific safe call
        row[f"{g}_call_level"] = p.get(f"{g}_call_level")          # allele|gene|family|none
        conf = p.get(f"{g}_set_confidence")
        row[f"{g}_set_confidence"] = f"{conf:.4f}" if conf is not None else None
    return row


class AirrWriter:
    """Incremental AIRR rearrangement TSV writer for bounded-memory streaming. Open once, call
    `write(ids, sequences, preds)` per chunk, then `close()` (or use as a context manager).
    `path` may be '-' for stdout."""

    def __init__(self, path: str, locus: str = "IGH", extra_columns=None):
        import sys
        self.locus = locus
        self.extra_columns = list(extra_columns or [])
        self._to_stdout = path == "-"
        self._f = sys.stdout if self._to_stdout else open(path, "w", newline="")
        fields = COLUMNS + [c for c in self.extra_columns if c not in COLUMNS]
        self._w = csv.DictWriter(self._f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        self._w.writeheader()

    def write(self, ids: List[str], sequences: List[str], preds: List[dict],
              metas: List[dict] | None = None) -> None:
        """`metas`, if given, is a per-row dict of extra column values (e.g. preserved barcode/UMI/
        sample metadata) merged into each output row."""
        for i, (sid, seq, p) in enumerate(zip(ids, sequences, preds)):
            row = _build_row(sid, seq, p, self.locus)
            if metas and metas[i]:
                row.update(metas[i])
            self._w.writerow(row)

    def close(self) -> None:
        if not self._to_stdout:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_airr(path: str, ids: List[str], sequences: List[str], preds: List[dict],
               locus: str = "IGH") -> None:
    """Eager one-shot write (back-compat). For large inputs use AirrWriter incrementally.
    `sequences` must be the CANONICAL (forward) sequences predict_reads' coordinates are in."""
    w = AirrWriter(path, locus)
    try:
        w.write(ids, sequences, preds)
    finally:
        w.close()
