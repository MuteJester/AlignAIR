"""AIRR rearrangement tables as the analysis commands read them: the one table reader, and the field parsers they
share."""
from __future__ import annotations

import csv
import os

_TRUE = frozenset({"t", "true", "1", "yes", "y"})


def read_airr_table(path: str | os.PathLike[str]) -> tuple[list[dict], list[str]]:
    """The rows of the AIRR table at ``path`` and its columns: tab-separated, or comma-separated when its header line
    holds no tab. Extra columns (AlignAIR's extensions among them) are kept."""
    with open(path, newline="") as stream:
        delimiter = "\t" if "\t" in stream.readline() else ","
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter=delimiter)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def truthy(value) -> bool:
    """An AIRR logical field's value: T, true, 1, yes or y (any case), or a bool."""
    if isinstance(value, bool):
        return value
    return value is not None and str(value).strip().lower() in _TRUE


def num(value) -> float | None:
    """A numeric field's value; None for a blank or a value that is not a number."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_call(call) -> str:
    """The first allele of a comma-separated call; empty for no call."""
    return str(call).split(",")[0].strip() if call else ""
