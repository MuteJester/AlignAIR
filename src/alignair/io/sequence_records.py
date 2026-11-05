"""Streaming sequence-file syntax, independent of alphabet validation and ID assignment."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import gzip
from itertools import chain
import os
import sys
from typing import NamedTuple
import zlib

from ..errors import ConfigurationError, InputError, OperationIOError

_SEQUENCE_COLUMNS = ("sequence", "seq", "nucleotide", "read")
_FORMATS = ("auto", "fasta", "fastq", "table", "txt")


class RawSequenceRecord(NamedTuple):
    """A source ID (possibly absent) and sequence text, without content validation."""
    sequence_id: str | None
    sequence: str


@contextmanager
def _open_text(path):
    if path == "-":
        yield sys.stdin
    else:
        opener = gzip.open if path.lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            yield stream


def detect_sequence_format(path: str, head: str, seq_column: str | None = None) -> str:
    """Prefer recognized suffixes, then the first nonblank line's syntax."""
    suffix = path.lower().removesuffix(".gz")
    if suffix.endswith((".fasta", ".fa", ".fna")):
        return "fasta"
    if suffix.endswith((".fastq", ".fq")):
        return "fastq"
    if suffix.endswith((".csv", ".tsv")):
        return "table"
    if head.startswith(">"):
        return "fasta"
    if head.startswith("@"):
        return "fastq"
    if any(delimiter in head for delimiter in (",", "\t")) and (
            seq_column is not None or any(name in head.lower() for name in _SEQUENCE_COLUMNS)):
        return "table"
    return "txt"


def _failure(message, path, fmt, line):
    return InputError(message, context={"stage": "sequence_file_parsing", "path": path,
                                        "format": fmt, "line": line})


def _fasta(lines, path):
    started, name, parts = False, None, []
    for number, line in lines:
        if line.startswith(">"):
            if started:
                yield RawSequenceRecord(name, "".join(parts))
            words = line[1:].split()
            started, name, parts = True, words[0] if words else None, []
        elif started:
            parts.append(line.rstrip("\r\n"))
        else:  # the source adapter has already skipped leading blank lines
            raise _failure("sequence occurs before its FASTA header", path, "fasta", number)
    yield RawSequenceRecord(name, "".join(parts))


def _fastq(lines, path):
    lines = iter(lines)
    for number, header in lines:
        if not header.startswith("@"):
            raise _failure(f"malformed FASTQ at line {number}: expected '@' header, "
                           f"got {header.rstrip()[:20]!r}", path, "fastq", number)
        sequence, separator, quality = next(lines, None), next(lines, None), next(lines, None)
        if quality is None:
            raise _failure(f"malformed FASTQ: truncated record starting at line {number} "
                           "(a record needs 4 lines)", path, "fastq", number)
        if not separator[1].startswith("+"):
            raise _failure(f"malformed FASTQ at line {separator[0]}: expected '+' separator, "
                           f"got {separator[1].rstrip()[:20]!r}", path, "fastq", separator[0])
        sequence, qualities = sequence[1].rstrip("\r\n"), quality[1].rstrip("\r\n")
        if len(sequence) != len(qualities):
            raise _failure(f"malformed FASTQ at line {quality[0]}: sequence length {len(sequence)} "
                           f"!= quality length {len(qualities)}", path, "fastq", quality[0])
        words = header[1:].split()
        yield RawSequenceRecord(words[0] if words else None, sequence)


def _table(lines, head, first_line, seq_column, id_column, path):
    delimiter = "\t" if path.lower().removesuffix(".gz").endswith(".tsv") or "\t" in head else ","
    reader = csv.DictReader((line for _, line in lines), delimiter=delimiter, strict=True)
    try:
        fields = reader.fieldnames or []
        if seq_column is not None and seq_column not in fields:
            raise _failure(f"sequence column '{seq_column}' not in {path} (have {fields})",
                           path, "table", first_line)
        column = seq_column if seq_column is not None else next((field for field in fields if field.lower() in _SEQUENCE_COLUMNS), None)
        if column is None:
            raise _failure(f"no sequence column in {path}; expected one of {_SEQUENCE_COLUMNS} "
                           "or select the sequence column explicitly", path, "table", first_line)
        if id_column is not None and id_column not in fields:
            raise _failure(f"id column '{id_column}' not in {path} (have {fields})",
                           path, "table", first_line)
        idcol = id_column if id_column is not None else next((field for field in fields if field.lower() in ("sequence_id", "id", "name")), None)
        for row in reader:
            if row[column] is None:
                raise _failure(f"missing sequence field in {path}", path, "table", first_line + reader.line_num - 1)
            yield RawSequenceRecord(row.get(idcol) if idcol else None, row[column])
    except csv.Error as exc:
        raise _failure(f"malformed sequence table: {exc}", path, "table", first_line + reader.reader.line_num - 1) from exc


def iter_sequence_records(path: str | os.PathLike[str], *, format: str = "auto",
                          seq_column: str | None = None, id_column: str | None = None):
    """Yield raw records from FASTA, four-line FASTQ, CSV/TSV or nonblank text lines.

    Plain/gzip paths and borrowed stdin (``-``) use one streaming pass. File handles close on
    exhaustion, errors or explicit iterator.close(); stdin remains open. FASTA emits empty and
    unnamed records; text ignores blank lines. IDs are header first words or table fields.
    Case, alphabet, duplicates and missing IDs are left to callers. FASTA/text sequence whitespace
    is retained (only line endings are dropped).
    """
    if not isinstance(format, str) or format not in _FORMATS:
        raise ConfigurationError(f"sequence format must be one of {_FORMATS}, got {format!r}")
    path = os.fspath(path)
    if not isinstance(path, str):
        raise ConfigurationError("sequence input path must be text")
    try:
        with _open_text(path) as stream:
            lines = enumerate(stream, 1)
            first = next((item for item in lines if item[1].strip()), None)
            if first is None:
                return
            number, head = first
            fmt = detect_sequence_format(path, head, seq_column) if format == "auto" else format
            lines = chain([first], lines)
            if fmt == "fasta":
                yield from _fasta(lines, path)
            elif fmt == "fastq":
                yield from _fastq(lines, path)
            elif fmt == "table":
                yield from _table(lines, head, number, seq_column, id_column, path)
            else:
                for _, line in lines:
                    if line.strip():
                        yield RawSequenceRecord(None, line.rstrip("\r\n"))
    except (UnicodeError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
        raise InputError("cannot decode sequence input", context={"stage": "sequence_file_parsing", "path": path}) from exc
    except OSError as exc:
        raise OperationIOError("cannot read sequence input", context={"stage": "sequence_file_parsing", "path": path}) from exc
