"""Write DNAlignAIR predictions as an AIRR rearrangement TSV (+ optional uncertainty/resolution
extension columns the AIRR schema does not cover: per-gene equivalence set, resolution level, and
set confidence). Only the equivalence set (``*_call_set``) is populated by default; the resolution
level and set confidence are reserved for the optional (separate) calibration step."""
from __future__ import annotations

import csv
import os
from typing import List

GENES = ("v", "d", "j")
# The full AIRR rearrangement schema we emit (a superset of what predict()+build_airr produce), then
# our optional uncertainty/resolution extensions (only `*_call_set` is populated by default). Per the
# AIRR `rev_comp` contract: a reverse-complement row (id 1) emits the ORIGINAL post-crop query as
# `sequence` with `rev_comp=T`, and its coordinates/alignments are on `RC(sequence)` (the model's
# forward frame); forward/complement/reverse rows emit the forward frame with `rev_comp=F`.
_IDENT = ["sequence_id", "sequence", "rev_comp", "locus"]
# c_call (constant-region gene) is a valid AIRR field the model does not predict but a side table often
# supplies (e.g. 10x c_gene). It is part of the schema so it is protected + fill-only (see below).
_CALLS = ["v_call", "d_call", "j_call", "c_call"]
_QUALITY = ["productive", "productive_prediction", "vj_in_frame", "stop_codon",
            "v_identity", "d_identity", "j_identity"]
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
# extension fields: full-transform orientation label + original (pre-orientation) input read +
# per-read segmentation quality flag. `orientation`/`input_sequence` keep the AIRR standard `rev_comp`
# boolean honest (it is set only for a true reverse-complement) without losing the actual transform.
_MISC = ["mutation_rate", "is_contaminant", "orientation", "input_sequence", "segmentation_low_quality",
         "length_cropped", "airr_assembly_status", "airr_assembly_reason", "airr_assembly_error"]
COLUMNS = (_IDENT + _CALLS + _QUALITY + _ALN + _JUNCTION + _CIGAR + _COORDS
           + _REGIONS + _PERGENE_ALN + _EXT + _MISC)

# orientation id -> AIRR-honest label (only id 1 is a reverse-complement; see _build_row)
_ORIENT_LABEL = {0: "forward", 1: "reverse_complement", 2: "complement", 3: "reverse"}

# model/scientific fields that per-read metadata must never overwrite: every produced
# AIRR column. Metadata may only add new columns or fill blanks; a collision keeps the model's value.
_PROTECTED_FIELDS = frozenset(COLUMNS)

# AIRR fields we WANT a side metadata table to be able to fill (the model does not produce them) but
# must never OVERWRITE if a value is already present: they keep their AIRR name (not namespaced to
# ``meta_*``) and, being protected, the writer's merge fills them only when the record's value is blank.
_METADATA_FILL_ONLY = frozenset({"c_call"})


class PredictionCountMismatch(RuntimeError):
    """The writer was given a different number of ids / sequences / predictions — a pipeline or model
    defect. Raised instead of silently truncating (and losing reads) via ``zip``."""

# Named column presets users can pick from (or pass their own list / comma-string). `full` is the
# default; `airr` is the MiAIRR-minimal required rearrangement set; `core`/`minimal` are compact.
COLUMN_PRESETS = {
    "full": list(COLUMNS),
    "core": (_IDENT + _CALLS + ["productive", "junction", "junction_aa", "junction_length"]
             + _COORDS + _CIGAR),
    "minimal": ["sequence_id", "sequence", "locus", "v_call", "d_call", "j_call", "productive"],
    "airr": ["sequence_id", "sequence", "rev_comp", "productive", "v_call", "d_call", "j_call",
             "sequence_alignment", "germline_alignment", "junction", "junction_aa",
             "v_cigar", "d_cigar", "j_cigar"],
}


def resolve_columns(columns) -> list:
    """Resolve a column selection to an ordered list of field names. ``columns`` may be:
    ``None`` (the full schema), a preset name (``full``/``core``/``minimal``/``airr``), a
    comma-separated string, or an explicit list of field names. Unknown names are allowed (emitted
    empty), so custom AIRR extension columns still pass through."""
    if columns is None:
        return list(COLUMNS)
    if isinstance(columns, str):
        if columns in COLUMN_PRESETS:
            return list(COLUMN_PRESETS[columns])
        return [c.strip() for c in columns.split(",") if c.strip()]
    return list(columns)


# fields that come straight from the light predict() record (no build_airr assembly needed)
_LIGHT_FIELDS = frozenset(_IDENT + _CALLS + ["productive", "productive_prediction", "mutation_rate",
                          "orientation", "input_sequence", "segmentation_low_quality", "length_cropped",
                          "airr_assembly_status", "airr_assembly_reason", "airr_assembly_error"]
                          + _CIGAR + _COORDS + _EXT)


def needs_assembly(columns) -> bool:
    """True if the selection includes any field only ``build_airr`` produces (junction / regions /
    alignments / identity); False when every selected field is a light-record field (calls / coords /
    cigar), so the caller can skip the AIRR assembly for speed."""
    return any(c not in _LIGHT_FIELDS for c in resolve_columns(columns))


def _airr_start(v):
    """0-based (GenAIRR/our convention) -> 1-based (AIRR)."""
    return (int(v) + 1) if v is not None else None


def _airr_bool(v):
    """Normalize a logical value to the AIRR ``T``/``F`` convention (None/blank stays empty/unknown)."""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        s = v.strip().upper()
        return "T" if s in ("T", "TRUE", "1", "YES") else ("F" if s in ("F", "FALSE", "0", "NO") else v)
    return "T" if v else "F"


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
    """Build one AIRR rearrangement row from a predict()/build_airr record.

    Coordinates/CIGAR/alignments are always computed in the model's forward frame (``p["sequence"]``).
    The emitted ``sequence`` field follows the AIRR convention for ``rev_comp``: for a
    reverse-complement read (id 1) ``sequence`` is the ORIGINAL query and
    ``rev_comp=T`` (coordinates apply to ``RC(sequence)`` == the forward frame, the IgBLAST/AIRR
    convention); for forward / complement-only / reverse-only, ``sequence`` is the forward frame with
    ``rev_comp=F`` (the true transform kept in the ``orientation`` extension, original in
    ``input_sequence``). So coordinates never require a double reverse.

    The "original query" is taken from the record itself (``p["input_sequence"]``, the post-crop
    pre-orientation read the pipeline stores) so it is crop-consistent and identical for the Python API
    and the CLI; the ``seq`` argument is only a fallback for records that predate that field.

    Applies the AIRR conventions: 1-based sequence/germline starts, ``rev_comp`` set only for a true
    reverse-complement (id 1), and
    coordinate-derived CIGAR / sequence_alignment fallbacks for records that skipped ``build_airr``."""
    row = dict(p)                                  # pass through all AIRR fields (junction/regions/...)
    # `canonical` is the frame every coordinate / alignment / CIGAR is computed in (the model's forward
    # frame). It is what all the slicing below uses. The emitted `sequence` field, however, follows the
    # AIRR convention for `rev_comp` (below), which is NOT always the canonical frame.
    canonical = p.get("sequence") if p.get("sequence") is not None else seq
    # the post-crop, pre-orientation read the RECORD owns (crop-consistent, path-independent); the
    # external `seq` is only a fallback for older records without it.
    original = p.get("input_sequence")
    if original is None or original == "":
        original = seq if seq is not None else canonical
    row.pop("input_sequence", None)                # re-set below per orientation (never emit canonical==)
    seq_len = len(canonical)
    row["sequence_id"] = sid
    row.setdefault("locus", locus)
    oid = int(p.get("orientation_id", 0) or 0)
    if oid == 1:
        # AIRR reverse-complement: `sequence` is the ORIGINAL query and, per the schema, all alignment
        # data (coordinates/CIGAR/alignments) are based on the REVERSE COMPLEMENT of `sequence` — which
        # is exactly our canonical frame. So emit the original + rev_comp=T (IgBLAST / AIRR-consumer
        # convention); a consumer reconstructs the aligned frame as RC(sequence) == canonical.
        row["sequence"] = original
        row["rev_comp"] = "T"
    else:
        # forward / complement-only / reverse-only: coordinates are on the emitted sequence directly, so
        # emit the canonical frame and rev_comp=F. `rev_comp` can only encode reverse-complement, so the
        # true transform for complement/reverse is preserved in the `orientation` extension.
        row["sequence"] = canonical
        row["rev_comp"] = "F"
        if original != canonical:                  # complement/reverse (or cropped): keep the original read
            row["input_sequence"] = original
    row["orientation"] = _ORIENT_LABEL.get(oid, "forward")
    # normalize AIRR logical fields to T/F (standards-compliant; passes official `airr` validation).
    # None/blank stays empty (unknown); the extensions are normalized too.
    for _lf in ("productive", "vj_in_frame", "stop_codon", "productive_prediction",
                "segmentation_low_quality", "length_cropped"):
        if row.get(_lf) is not None and row.get(_lf) != "":
            row[_lf] = _airr_bool(row[_lf])
    isc = p.get("is_contaminant")
    row["is_contaminant"] = ("T" if isc else "F") if isc is not None else None
    if not row.get("sequence_alignment"):          # fallback when build_airr was not run
        starts = [p.get(f"{g}_sequence_start") for g in GENES if p.get(f"{g}_sequence_start") is not None]
        ends = [p.get(f"{g}_sequence_end") for g in GENES if p.get(f"{g}_sequence_end")]
        row["sequence_alignment"] = canonical[min(starts):max(ends)] if starts and ends else ""
    for g in GENES:
        ss, se = p.get(f"{g}_sequence_start"), p.get(f"{g}_sequence_end")
        # An absent / zero-length segment (e.g. Short-D) has no span; AIRR 1-based-inclusive coords
        # cannot encode an empty interval (start would exceed end), so blank its coords + CIGAR.
        if ss is not None and se is not None and se <= ss:
            row[f"{g}_cigar"] = ""
            row[f"{g}_sequence_start"] = row[f"{g}_sequence_end"] = None
            row[f"{g}_germline_start"] = row[f"{g}_germline_end"] = None
        else:
            row[f"{g}_cigar"] = p.get(f"{g}_cigar") or _cigar(
                seq_len, ss, se, p.get(f"{g}_germline_start"), p.get(f"{g}_germline_end"))
            row[f"{g}_sequence_start"] = _airr_start(ss)                # 0-based -> 1-based
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

    def __init__(self, path: str, locus: str = "IGH", columns=None, extra_columns=None):
        """``columns``: which fields to emit (a preset name, comma-string, or list; default = the full
        schema). ``extra_columns``: extra per-row metadata columns (barcode/UMI/sample) appended after.

        A file output is written **atomically**: rows go to a sibling temp file that is renamed onto
        ``path`` only on a clean ``close``/context exit, so an interrupted job never leaves a final path
        that looks complete."""
        import sys
        self.locus = locus
        self.extra_columns = list(extra_columns or [])
        self._to_stdout = path == "-"
        self._final_path = None if self._to_stdout else path
        self._tmp_path = None if self._to_stdout else f"{path}.tmp.{os.getpid()}"
        self._f = sys.stdout if self._to_stdout else open(self._tmp_path, "w", newline="")
        base = resolve_columns(columns)
        fields = base + [c for c in self.extra_columns if c not in base]
        self._w = csv.DictWriter(self._f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        self._w.writeheader()

    def write(self, ids: List[str], sequences: List[str], preds: List[dict],
              metas: List[dict] | None = None) -> None:
        """`sequences` are the ORIGINAL input reads; the emitted `sequence`/`rev_comp` follow the AIRR
        orientation convention per record. `metas`, if given, is a per-row dict of extra column values
        (barcode/UMI/sample/cell_id) merged into each output row without overwriting model fields.

        The three lists must be the same length — a mismatch (a pipeline/model defect returning fewer or
        extra predictions) raises :class:`PredictionCountMismatch` rather than silently truncating via
        ``zip`` and losing reads."""
        if not (len(ids) == len(sequences) == len(preds)):
            raise PredictionCountMismatch(
                f"cardinality mismatch: ids={len(ids)}, sequences={len(sequences)}, "
                f"predictions={len(preds)} — refusing to write (would silently drop reads)")
        if metas is not None and len(metas) != len(ids):
            raise PredictionCountMismatch(f"metas={len(metas)} != records={len(ids)}")
        for i, (sid, seq, p) in enumerate(zip(ids, sequences, preds)):
            row = _build_row(sid, seq, p, self.locus)
            if metas and metas[i]:
                for k, v in metas[i].items():      # metadata may NOT clobber a populated model field
                    if k in _PROTECTED_FIELDS and row.get(k) not in (None, ""):
                        continue
                    row[k] = v
            self._w.writerow(row)

    def close(self, commit: bool = True) -> None:
        """Close the output. For a file target, ``commit=True`` atomically renames the temp file onto
        the final path; ``commit=False`` discards it (used on an aborted context exit)."""
        if self._to_stdout:
            return
        if not self._f.closed:
            self._f.close()
        if commit:
            os.replace(self._tmp_path, self._final_path)
        elif os.path.exists(self._tmp_path):
            try:
                os.remove(self._tmp_path)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(commit=exc[0] is None)          # never commit a partial file on an exception


def write_airr(path: str, ids: List[str], sequences: List[str], preds: List[dict],
               locus: str = "IGH", columns=None, metas: List[dict] | None = None,
               extra_columns=None) -> None:
    """Eager one-shot write (back-compat). For large inputs use AirrWriter incrementally.
    `sequences` are the ORIGINAL input reads; the emitted `sequence`/`rev_comp` follow the AIRR
    orientation convention per record. `columns` selects which fields to emit. `metas` (per-row
    dicts) + `extra_columns` carry preserved input metadata (barcode/UMI/sample/cell_id) into output."""
    with AirrWriter(path, locus, columns=columns, extra_columns=extra_columns) as w:  # atomic
        w.write(ids, sequences, preds, metas=metas)
