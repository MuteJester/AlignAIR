"""Read input reads from common formats for `alignair predict`.

Supports FASTA, FASTQ, CSV/TSV (a sequence column), and plain one-per-line TXT, optionally
gzip-compressed. Returns (ids, sequences) with sequences uppercased and validated: IUPAC
ambiguity codes -> N; a read with >20% non-ACGTN characters is dropped (reported).
"""
from __future__ import annotations

import csv
import gzip
import io
import os
from typing import List, Tuple

_VALID = set("ACGTN")
_IUPAC = set("RYSWKMBDHVN")          # ambiguity codes -> N
_SEQ_COLS = ("sequence", "seq", "nucleotide", "read")


def _open_stream(path: str):
    """Open ``path`` for a SINGLE streaming pass. stdin (``-``) is returned as-is — never slurped or
    cached — so a piped 1M-read repertoire streams in bounded memory."""
    if path == "-":
        import sys
        return sys.stdin
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r")


def validate_sequence(seq: str, max_len: int | None = None) -> tuple[str | None, str | None]:
    """The single content validator (used by the reader and the predict pipeline).

    Returns ``(cleaned, reason)``: ``cleaned`` is the uppercased sequence with IUPAC ambiguity codes
    mapped to ``N``; ``reason`` is ``None`` when accepted, else a machine-readable rejection code:
    ``"empty"`` (blank), ``"ambiguous"`` (>20% unusable characters), or ``"too_long"`` (exceeds
    ``max_len`` — callers decide crop vs reject; the reader never silently truncates)."""
    s = seq.strip().upper()
    if not s:
        return None, "empty"
    bad = 0
    out = []
    for c in s:
        if c in _VALID:
            out.append(c)
        elif c in _IUPAC:
            out.append("N")
        else:
            bad += 1
            out.append("N")
    if bad / len(s) > 0.20:
        return None, "ambiguous"
    cleaned = "".join(out)
    if max_len is not None and len(cleaned) > max_len:
        return None, "too_long"
    return cleaned, None


def validate(seq: str) -> str | None:
    """Back-compat wrapper: the cleaned sequence, or ``None`` if unusable (see ``validate_sequence``)."""
    return validate_sequence(seq)[0]


def _sniff(path: str, head: str, seq_column: str | None = None) -> str:
    ext = os.path.basename(path).lower().rstrip(".gz") if False else os.path.basename(path).lower()
    ext = ext[:-3] if ext.endswith(".gz") else ext
    if ext.endswith((".fasta", ".fa", ".fna")):
        return "fasta"
    if ext.endswith((".fastq", ".fq")):
        return "fastq"
    if ext.endswith((".csv", ".tsv")):
        return "table"
    if head.startswith(">"):
        return "fasta"
    if head.startswith("@"):
        return "fastq"
    # extensionless (e.g. piped stdin): a delimited header is a table if it names a known sequence
    # column OR the caller explicitly told us the column via --sequence-column.
    if any(c in head for c in (",", "\t")) and (
            seq_column is not None or any(k in head.lower() for k in _SEQ_COLS)):
        return "table"
    return "txt"


# metadata columns worth carrying into output by default (10x / AIRR / Immcantation single-cell)
_META_DEFAULTS = ["cell_id", "barcode", "sample_id", "umi_count", "umis", "reads",
                  "duplicate_count", "consensus_count", "raw_clonotype_id", "raw_consensus_id",
                  "chain", "c_call", "is_cell", "high_confidence"]
# 10x Cell Ranger contig-annotation columns -> their AIRR-standard names (so Scirpy/AIRR consumers get
# `cell_id`/`umi_count`/`c_call`, not just raw 10x columns). The raw columns are preserved as well.
_10X_TO_AIRR = {"barcode": "cell_id", "umis": "umi_count", "reads": "consensus_count", "c_gene": "c_call"}


def _metadata_plan(fields, id_column, keep_columns, normalize_10x, path):
    """Resolve (id_column, keep_columns, 10x-normalization map, output columns) from the header."""
    idcol = id_column or next(
        (c for c in fields if c.lower() in ("sequence_id", "contig_id", "cell_id", "id", "name")), None)
    if idcol is None or idcol not in fields:
        raise ValueError(f"metadata id column {id_column or '(auto)'} not found in {path} "
                         f"(columns: {fields})")
    if keep_columns:
        missing = [c for c in keep_columns if c not in fields]
        if missing:
            raise ValueError(f"--keep-columns not in {path}: {missing} (have {fields})")
        keep = [c for c in keep_columns if c in fields]
    else:
        keep = [c for c in _META_DEFAULTS if c in fields]
    norm = {src: tgt for src, tgt in _10X_TO_AIRR.items() if normalize_10x and src in fields}
    out_cols = list(keep) + [tgt for tgt in norm.values() if tgt not in keep]
    return idcol, keep, norm, out_cols


def _iter_metadata_rows(path, id_column, keep_columns, normalize_10x):
    """Yield (id, normalized_row) from a metadata table; second yield-item exposes out_cols via the
    generator's ``.out_cols`` set after the first row. Streams — never holds the whole table."""
    with open(path, newline="") as f:
        delim = "\t" if "\t" in f.readline() else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        idcol, keep, norm, out_cols = _metadata_plan(reader.fieldnames or [], id_column, keep_columns,
                                                     normalize_10x, path)
        yield ("__cols__", out_cols)               # header sentinel first
        for r in reader:
            rid = r.get(idcol)
            if not rid:
                continue
            row = {k: r.get(k, "") for k in keep}
            for src, tgt in norm.items():          # add the AIRR-normalized name (raw src kept too)
                if r.get(src):
                    row[tgt] = r[src]
            yield rid, row


def load_metadata(path: str, id_column: str | None = None, keep_columns=None, normalize_10x: bool = False):
    """Load a per-read metadata table (CSV/TSV, e.g. 10x filtered_contig_annotations.csv or an AIRR
    TSV) -> ({read_id: {col: value}}, kept_columns) — the in-memory form (fine for tests / small
    tables). For a large repertoire, prefer :func:`build_metadata_index` (disk-backed)."""
    it = _iter_metadata_rows(path, id_column, keep_columns, normalize_10x)
    out_cols = next(it)[1]
    return {rid: row for rid, row in it}, out_cols


class DuplicateMetadataId(ValueError):
    """The metadata join key is not unique (an id appears on more than one row), which would make the
    per-read join ambiguous. Raised so the caller can pick a unique --metadata-id-column instead of one
    row silently winning."""


# SQLite caps host parameters per statement (SQLITE_MAX_VARIABLE_NUMBER; historically 999). Stay well
# under it so an ``id IN (?, ?, …)`` lookup never overflows for a large prediction chunk.
_SQL_PARAM_CHUNK = 900


class MetadataIndex:
    """A disk-backed (SQLite) per-read metadata lookup, so joining a large ``--metadata`` table does not
    consume memory proportional to the repertoire. Look reads up a CHUNK at a time with
    :meth:`get_many` (one indexed query per prediction batch, not one query per read)."""

    def __init__(self):
        import os as _os
        import sqlite3
        import tempfile
        fd, self._path = tempfile.mkstemp(suffix=".alignair-meta.sqlite")  # secure (not insecure mktemp)
        _os.close(fd)                                  # sqlite opens its own handle to the path
        self._con = sqlite3.connect(self._path)
        self._con.execute("CREATE TABLE m (id TEXT PRIMARY KEY, row TEXT)")

    def put_many(self, items) -> None:
        """Insert (id, row) pairs. A duplicate id (within this batch or vs. an earlier one) raises
        :class:`DuplicateMetadataId` naming the offending id — the join must be unambiguous, so we do
        NOT silently keep the first/last row."""
        import json
        import sqlite3
        rows = [(k, json.dumps(v)) for k, v in items]
        try:
            self._con.executemany("INSERT INTO m VALUES (?, ?)", rows)
            self._con.commit()
        except sqlite3.IntegrityError:
            self._con.rollback()                       # find the offending id (error path only)
            for k, payload in rows:
                try:
                    self._con.execute("INSERT INTO m VALUES (?, ?)", (k, payload))
                except sqlite3.IntegrityError:
                    self._con.rollback()
                    raise DuplicateMetadataId(
                        f"duplicate metadata id {k!r}; pass --metadata-id-column to choose a unique key"
                    ) from None
            self._con.commit()

    def get_many(self, ids) -> dict:
        """Return ``{id: row}`` for the ids present, in one indexed query per <=900-id sub-batch."""
        import json
        ids = list(ids)
        out: dict = {}
        for i in range(0, len(ids), _SQL_PARAM_CHUNK):
            batch = ids[i:i + _SQL_PARAM_CHUNK]
            q = "SELECT id, row FROM m WHERE id IN (%s)" % ",".join("?" * len(batch))
            for rid, payload in self._con.execute(q, batch):
                out[rid] = json.loads(payload)
        return out

    def get(self, key, default=None):
        return self.get_many([key]).get(key, {} if default is None else default)

    def close(self) -> None:
        import os as _os
        try:
            self._con.close()
        finally:
            try:
                _os.remove(self._path)
            except OSError:
                pass


def build_metadata_index(path, id_column=None, keep_columns=None, normalize_10x=False, rename=None):
    """Stream a metadata table into a disk-backed :class:`MetadataIndex` (bounded memory). ``rename(col)``
    optionally maps output column names (collision protection). Returns ``(index, out_columns)``. If the
    stream fails midway (e.g. a duplicate id), the temp database is closed and deleted before re-raising
    so no orphan file is left behind."""
    it = _iter_metadata_rows(path, id_column, keep_columns, normalize_10x)
    out_cols = next(it)[1]
    if rename:
        out_cols = [rename(c) for c in out_cols]
    idx = MetadataIndex()
    try:
        batch = []
        for rid, row in it:
            batch.append((rid, {rename(k): v for k, v in row.items()} if rename else row))
            if len(batch) >= 5000:
                idx.put_many(batch)
                batch = []
        if batch:
            idx.put_many(batch)
    except BaseException:
        idx.close()                                    # no orphan temp DB on failure
        raise
    return idx, out_cols


def _detect_format(path: str) -> str:
    if path == "-":                       # can't peek stdin without consuming its single pass
        return "stdin"
    with _open_stream(path) as f:
        head = f.readline()
    return _sniff(path, head)


def _iter_records(fmt: str, lines, head: str, seq_column, id_column, path: str):
    """Yield (id_or_None, raw_sequence) from a single line iterator (`head` already chained in) — never
    reopens or slurps the source, so stdin streams in bounded memory. `path` is used only for messages
    and the table delimiter."""
    if fmt == "fasta":
        name, buf = None, []
        for line in lines:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                tok = line[1:].split()
                name, buf = (tok[0] if tok else None), []
            else:
                buf.append(line)
        if name is not None:
            yield name, "".join(buf)
    elif fmt == "fastq":
        it = iter(lines)
        ln = 0
        while True:
            h = next(it, "")
            if not h:
                break
            ln += 1
            if not h.startswith("@"):
                raise ValueError(f"malformed FASTQ at line {ln}: expected '@' header, "
                                 f"got {h.rstrip()[:20]!r}")
            s, plus, q = next(it, ""), next(it, ""), next(it, "")
            if not q:
                raise ValueError(f"malformed FASTQ: truncated record starting at line {ln} "
                                 f"(a record needs 4 lines)")
            if not plus.startswith("+"):
                raise ValueError(f"malformed FASTQ at line {ln + 2}: expected '+' separator, "
                                 f"got {plus.rstrip()[:20]!r}")
            seq, qual = s.rstrip("\n"), q.rstrip("\n")
            if len(seq) != len(qual):
                raise ValueError(f"malformed FASTQ at line {ln + 3}: sequence length {len(seq)} "
                                 f"!= quality length {len(qual)}")
            ln += 3
            tok = h.rstrip("\n")[1:].split()
            yield (tok[0] if tok else None), seq
    elif fmt == "table":
        delim = "\t" if path.lower().rstrip(".gz").endswith(".tsv") or "\t" in head else ","
        reader = csv.DictReader(lines, delimiter=delim)
        fields = reader.fieldnames or []
        if seq_column is not None:
            if seq_column not in fields:
                raise ValueError(f"sequence column '{seq_column}' not in {path} (have {fields})")
            col = seq_column
        else:
            col = next((c for c in fields if c.lower() in _SEQ_COLS), None)
        if col is None:
            raise ValueError(f"no sequence column in {path}; expected one of {_SEQ_COLS} "
                             f"or pass --sequence-column")
        if id_column is not None:
            if id_column not in fields:
                raise ValueError(f"id column '{id_column}' not in {path} (have {fields})")
            idcol = id_column
        else:
            idcol = next((c for c in fields if c.lower() in ("sequence_id", "id", "name")), None)
        for row in reader:
            yield (row.get(idcol) if idcol else None), row.get(col, "")
    else:  # txt: one sequence per line
        for line in lines:
            if line.strip():
                yield None, line


def iter_sequences(path: str, chunk_size: int = 20000, seq_column: str | None = None,
                   id_column: str | None = None, rejects: list | None = None):
    """Stream (ids, sequences, n_dropped) in chunks of up to ``chunk_size`` validated reads, with
    bounded memory (never materializes the whole file). Sequences are validated as in
    ``read_sequences``; ids default to ``seq{global_index}`` when the source has none. If ``rejects``
    (a list) is given, each dropped read is appended as ``{id, position, reason, sequence}`` so the
    caller can emit a rejects table rather than silently dropping records."""
    import itertools
    stream = _open_stream(path)           # opened ONCE; stdin is not slurped
    ids: List[str] = []
    seqs: List[str] = []
    dropped = 0
    seen: dict = {}                       # base id -> collision count (persists across chunks)

    def _unique_id(rid, i):
        base = rid if rid is not None else f"seq{i}"
        n = seen.get(base)
        if n is None:                     # first occurrence keeps its id; preserve input order
            seen[base] = 0
            return base
        seen[base] = n + 1                # deterministic disambiguation of duplicate ids
        return f"{base}_dup{n + 1}"

    try:
        head = stream.readline()
        fmt = _sniff(path, head, seq_column)      # --sequence-column disambiguates a piped table
        lines = itertools.chain([head], stream)   # put the sniffed line back — no reopen, no slurp
        for i, (rid, raw) in enumerate(_iter_records(fmt, lines, head, seq_column, id_column, path)):
            v, reason = validate_sequence(raw)
            if v is None:
                dropped += 1
                if rejects is not None:
                    rejects.append({"sequence_id": rid if rid is not None else f"seq{i}", "position": i,
                                    "reason": reason, "sequence": str(raw).strip()})
            else:
                ids.append(_unique_id(rid, i))
                seqs.append(v)
            if len(seqs) >= chunk_size:
                yield ids, seqs, dropped
                ids, seqs, dropped = [], [], 0
    finally:
        if path != "-":                   # leave stdin open for the parent process
            stream.close()
    if ids or dropped:
        yield ids, seqs, dropped


def read_sequences(path: str, seq_column: str | None = None, id_column: str | None = None,
                   collect_rejects: bool = False) -> Tuple[List[str], List[str], dict]:
    """Eager read of all sequences (back-compat). For large files prefer ``iter_sequences``.
    Returns (ids, sequences, info) with info.n_read/n_dropped/format; when ``collect_rejects`` is set,
    info["rejects"] lists each dropped record ({id, position, reason, sequence})."""
    ids: List[str] = []
    seqs: List[str] = []
    dropped = 0
    rejects: list | None = [] if collect_rejects else None
    for cids, cseqs, drp in iter_sequences(path, chunk_size=10 ** 9, seq_column=seq_column,
                                           id_column=id_column, rejects=rejects):
        ids += cids; seqs += cseqs; dropped += drp
    info = {"n_read": len(ids) + dropped, "n_dropped": dropped, "format": _detect_format(path)}
    if collect_rejects:
        info["rejects"] = rejects
    return ids, seqs, info
