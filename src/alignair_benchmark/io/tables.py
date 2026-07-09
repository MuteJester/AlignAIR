"""Delimited-table helpers for benchmark prediction ingestion."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..evaluation import igblast_airr_to_prediction


def _resolve_delimiter(path: Path, delimiter: str | None) -> str:
    if delimiter is not None:
        return delimiter.encode("utf-8").decode("unicode_escape")
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"
    if path.suffix.lower() == ".csv":
        return ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
    except csv.Error:
        return "\t"


def load_table_rows(path: str | Path, *, delimiter: str | None = None) -> list[dict[str, Any]]:
    """Load a delimited table into a list of dictionaries.

    The delimiter is inferred from ``.tsv``/``.tab``/``.csv`` suffixes, then by
    ``csv.Sniffer`` for extensionless files. Pass ``delimiter="\\t"`` to force
    tab parsing from shells that cannot conveniently pass a literal tab.
    """

    path = Path(path)
    resolved = _resolve_delimiter(path, delimiter)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=resolved)
        return [dict(row) for row in reader]


def load_airr_predictions(path: str | Path, *, delimiter: str | None = None) -> list[dict[str, Any]]:
    """Load AIRR/IgBLAST-style rows and normalize them for benchmark scoring."""

    return [igblast_airr_to_prediction(row) for row in load_table_rows(path, delimiter=delimiter)]
