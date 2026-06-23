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


_STDIN_CACHE = None


def _slurp_stdin() -> str:
    global _STDIN_CACHE
    if _STDIN_CACHE is None:
        import sys
        _STDIN_CACHE = sys.stdin.read()
    return _STDIN_CACHE


def _open(path: str):
    if path == "-":                       # read from stdin (cached so it can be re-opened)
        return io.StringIO(_slurp_stdin())
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r")


def validate(seq: str) -> str | None:
    """Uppercase, map IUPAC ambiguity -> N; return None if >20% chars are unusable."""
    s = seq.strip().upper()
    if not s:
        return None
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
        return None
    return "".join(out)


def _sniff(path: str, head: str) -> str:
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
    if any(c in head for c in (",", "\t")) and any(k in head.lower() for k in _SEQ_COLS):
        return "table"
    return "txt"


# metadata columns worth carrying into output by default (10x / AIRR / Immcantation single-cell)
_META_DEFAULTS = ["cell_id", "barcode", "sample_id", "umi_count", "umis", "reads",
                  "duplicate_count", "consensus_count", "raw_clonotype_id", "raw_consensus_id",
                  "chain", "c_call", "is_cell", "high_confidence"]


def load_metadata(path: str, id_column: str | None = None, keep_columns=None):
    """Load a per-read metadata table (CSV/TSV, e.g. 10x filtered_contig_annotations.csv or an AIRR
    TSV) -> ({read_id: {col: value}}, kept_columns). The id column is matched to the read ids in the
    input; `keep_columns` (or a default 10x/AIRR set present in the file) are carried into output."""
    with open(path, newline="") as f:
        delim = "\t" if "\t" in f.readline() else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        fields = reader.fieldnames or []
        idcol = id_column or next(
            (c for c in fields if c.lower() in ("sequence_id", "contig_id", "cell_id", "id", "name")), None)
        if idcol is None or idcol not in fields:
            raise ValueError(f"metadata id column {id_column or '(auto)'} not found in {path} "
                             f"(columns: {fields})")
        if keep_columns:
            keep = [c for c in keep_columns if c in fields]
            missing = [c for c in keep_columns if c not in fields]
            if missing:
                raise ValueError(f"--keep-columns not in {path}: {missing} (have {fields})")
        else:
            keep = [c for c in _META_DEFAULTS if c in fields]
        meta = {r[idcol]: {k: r.get(k, "") for k in keep} for r in reader if r.get(idcol)}
    return meta, keep


def _detect_format(path: str) -> str:
    with _open(path) as f:
        head = f.readline()
    return _sniff(path, head)


def _iter_records(path: str, fmt: str, head: str, seq_column, id_column):
    """Yield (id_or_None, raw_sequence) LAZILY for one file/stdin — never loads the whole file."""
    if fmt == "fasta":
        name, buf = None, []
        with _open(path) as f:
            for line in f:
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
        with _open(path) as f:
            while True:
                h = f.readline()
                if not h:
                    break
                s = f.readline(); f.readline(); f.readline()      # seq, '+', qual
                if h.startswith("@"):
                    tok = h.rstrip("\n")[1:].split()
                    yield (tok[0] if tok else None), s.rstrip("\n")
    elif fmt == "table":
        delim = "\t" if path.lower().rstrip(".gz").endswith(".tsv") or "\t" in head else ","
        with _open(path) as f:
            reader = csv.DictReader(f, delimiter=delim)
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
        with _open(path) as f:
            for line in f:
                if line.strip():
                    yield None, line


def iter_sequences(path: str, chunk_size: int = 20000, seq_column: str | None = None,
                   id_column: str | None = None):
    """Stream (ids, sequences, n_dropped) in chunks of up to ``chunk_size`` validated reads, with
    bounded memory (never materializes the whole file). Sequences are validated as in
    ``read_sequences``; ids default to ``seq{global_index}`` when the source has none."""
    with _open(path) as f:
        head = f.readline()
    fmt = _sniff(path, head)
    ids: List[str] = []
    seqs: List[str] = []
    dropped = 0
    for i, (rid, raw) in enumerate(_iter_records(path, fmt, head, seq_column, id_column)):
        v = validate(raw)
        if v is None:
            dropped += 1
        else:
            ids.append(rid if rid is not None else f"seq{i}")
            seqs.append(v)
        if len(seqs) >= chunk_size:
            yield ids, seqs, dropped
            ids, seqs, dropped = [], [], 0
    if ids or dropped:
        yield ids, seqs, dropped


def read_sequences(path: str, seq_column: str | None = None,
                   id_column: str | None = None) -> Tuple[List[str], List[str], dict]:
    """Eager read of all sequences (back-compat). For large files prefer ``iter_sequences``.
    Returns (ids, sequences, info) with info.n_read/n_dropped/format."""
    ids: List[str] = []
    seqs: List[str] = []
    dropped = 0
    for cids, cseqs, drp in iter_sequences(path, chunk_size=10 ** 9,
                                           seq_column=seq_column, id_column=id_column):
        ids += cids; seqs += cseqs; dropped += drp
    return ids, seqs, {"n_read": len(ids) + dropped, "n_dropped": dropped,
                       "format": _detect_format(path)}
