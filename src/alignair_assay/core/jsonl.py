"""JSON Lines files with an optional header line; a ``.gz`` path is gzip-compressed.

A file may start with a header line ``{"header": {...}}`` describing how it was built; ``read_jsonl`` skips it and
``read_header`` returns it. Rows are written compactly and as strict JSON (a NaN or infinity is an error).
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path


def write_jsonl(rows: Iterable[dict], path: str | Path, header: dict | None = None) -> None:
    """Write ``rows`` one JSON object per line, after the header line when ``header`` is given."""
    with _open(path, "wt") as fh:
        if header is not None:
            fh.write(_dumps({"header": header}) + "\n")
        for row in rows:
            fh.write(_dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Every row of the file, without its header line; blank lines are skipped."""
    with _open(path, "rt") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[1:] if rows and _is_header(rows[0]) else rows


def read_header(path: str | Path) -> dict | None:
    """The header of the file, or None when its first row is not a header line."""
    with _open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                return row["header"] if _is_header(row) else None
    return None


def _is_header(row) -> bool:
    return isinstance(row, dict) and set(row) == {"header"}


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def _open(path: str | Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")
