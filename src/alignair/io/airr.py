"""Write DNAlignAIR predictions as an AIRR rearrangement TSV (+ calibrated-uncertainty
extension columns the AIRR schema does not cover: per-gene equivalence set, resolution
level, and set confidence)."""
from __future__ import annotations

import csv
from typing import List

GENES = ("v", "d", "j")
# The full AIRR rearrangement schema we emit (a superset of what predict()+build_airr produce),
# then our calibrated-uncertainty extensions. `sequence` is the CANONICAL (forward) sequence the
# coordinates refer to; `rev_comp` flags that the input read was reoriented to produce it.
_IDENT = ["sequence_id", "sequence", "rev_comp", "locus"]
_CALLS = ["v_call", "d_call", "j_call"]
_QUALITY = ["productive", "vj_in_frame", "stop_codon", "v_identity", "d_identity", "j_identity"]
_ALN = ["sequence_alignment", "germline_alignment", "sequence_alignment_aa", "germline_alignment_aa"]
_JUNCTION = ["junction", "junction_aa", "junction_length", "junction_aa_length",
             "np1", "np1_length", "np2", "np2_length"]
_CIGAR = [f"{g}_cigar" for g in GENES]
_COORDS = [f"{g}_{k}" for g in GENES
           for k in ("sequence_start", "sequence_end", "germline_start", "germline_end")]
_REGIONS = [f"{r}{suf}" for r in ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")
            for suf in ("", "_aa", "_start", "_end")]
_PERGENE_ALN = [f"{g}_{k}" for g in GENES
                for k in ("sequence_alignment", "germline_alignment",
                          "sequence_alignment_aa", "germline_alignment_aa",
                          "alignment_start", "alignment_end")]
_EXT = [f"{g}_{k}" for g in GENES for k in ("call_set", "resolved_call", "call_level", "set_confidence")]
_MISC = ["mutation_rate", "is_contaminant"]
COLUMNS = (_IDENT + _CALLS + _QUALITY + _ALN + _JUNCTION + _CIGAR + _COORDS
           + _REGIONS + _PERGENE_ALN + _EXT + _MISC)


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
    """Build one AIRR rearrangement row from a predict()/build_airr record. `seq` is the CANONICAL
    (forward) sequence the coordinates are in. Passes the record's AIRR fields through, applying the
    AIRR conventions: 1-based sequence/germline starts, the ``rev_comp`` flag, and coordinate-derived
    CIGAR / sequence_alignment fallbacks for records that did not go through ``build_airr``."""
    row = dict(p)                                  # pass through all AIRR fields (junction/regions/...)
    seq_len = len(seq)
    row["sequence_id"] = sid
    row["sequence"] = seq
    row.setdefault("locus", locus)
    row["rev_comp"] = "T" if p.get("orientation_id", 0) != 0 else "F"
    isc = p.get("is_contaminant")
    row["is_contaminant"] = ("T" if isc else "F") if isc is not None else None
    if not row.get("sequence_alignment"):          # fallback when build_airr was not run
        starts = [p.get(f"{g}_sequence_start") for g in GENES if p.get(f"{g}_sequence_start") is not None]
        ends = [p.get(f"{g}_sequence_end") for g in GENES if p.get(f"{g}_sequence_end")]
        row["sequence_alignment"] = seq[min(starts):max(ends)] if starts and ends else ""
    for g in GENES:
        row[f"{g}_cigar"] = p.get(f"{g}_cigar") or _cigar(
            seq_len, p.get(f"{g}_sequence_start"), p.get(f"{g}_sequence_end"),
            p.get(f"{g}_germline_start"), p.get(f"{g}_germline_end"))
        row[f"{g}_sequence_start"] = _airr_start(p.get(f"{g}_sequence_start"))   # 0-based -> 1-based
        row[f"{g}_germline_start"] = _airr_start(p.get(f"{g}_germline_start"))
        cset = (p.get(f"{g}_call_set") or p.get(f"{g}_calls")
                or ([p.get(f"{g}_call")] if p.get(f"{g}_call") else []))
        row[f"{g}_call_set"] = ",".join(c for c in cset if c)
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
