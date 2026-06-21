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


def _open(path: str):
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


def read_sequences(path: str) -> Tuple[List[str], List[str], dict]:
    """Return (ids, sequences, info). info has n_read/n_dropped for reporting."""
    with _open(path) as f:
        head = f.readline()
    fmt = _sniff(path, head)
    ids: List[str] = []
    raw: List[str] = []
    if fmt == "fasta":
        name, buf = None, []
        with _open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if name is not None:
                        ids.append(name); raw.append("".join(buf))
                    name, buf = line[1:].split()[0] or f"seq{len(ids)}", []
                else:
                    buf.append(line)
            if name is not None:
                ids.append(name); raw.append("".join(buf))
    elif fmt == "fastq":
        with _open(path) as f:
            lines = [ln.rstrip("\n") for ln in f]
        for i in range(0, len(lines) - 3, 4):
            if lines[i].startswith("@"):
                ids.append(lines[i][1:].split()[0]); raw.append(lines[i + 1])
    elif fmt == "table":
        delim = "\t" if path.lower().rstrip(".gz").endswith(".tsv") or "\t" in head else ","
        with _open(path) as f:
            reader = csv.DictReader(f, delimiter=delim)
            col = next((c for c in reader.fieldnames or [] if c.lower() in _SEQ_COLS), None)
            if col is None:
                raise ValueError(f"no sequence column in {path}; expected one of {_SEQ_COLS}")
            idcol = next((c for c in reader.fieldnames if c.lower() in ("sequence_id", "id", "name")), None)
            for i, row in enumerate(reader):
                ids.append(row.get(idcol) if idcol else f"seq{i}"); raw.append(row.get(col, ""))
    else:  # txt: one sequence per line
        with _open(path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    ids.append(f"seq{i}"); raw.append(line)

    seqs, kept_ids, dropped = [], [], 0
    for sid, s in zip(ids, raw):
        v = validate(s)
        if v is None:
            dropped += 1
        else:
            kept_ids.append(sid); seqs.append(v)
    return kept_ids, seqs, {"n_read": len(raw), "n_dropped": dropped, "format": fmt}
