"""CSV/TSV reader producing per-row dicts; defaults missing optional columns.

Uses the standard library, including for gzip/bzip2/xz-compressed tables. Values stay
as strings; the record adapter handles numeric and boolean conversion.
"""
import bz2
import csv
import gzip
from itertools import islice
import logging
import lzma
from pathlib import Path

from ..errors import ConfigurationError, InputError

from .column_schema import ColumnSet

logger = logging.getLogger(__name__)

_DEFAULTS = {"productive": 1.0, "indels": ""}


class CsvTableReader:
    def __init__(self, path: str, column_set: ColumnSet, sep: str = ",",
                 nrows: int | None = None):
        if nrows is not None and (not isinstance(nrows, int) or nrows < 0):
            raise ConfigurationError("nrows must be a non-negative integer or None", context={"nrows": nrows})
        self.columns = column_set.as_list()
        self.records = []
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(Path(path).suffix, open)
        # newline='' preserves CSV quoting/newlines; utf-8-sig also accepts a UTF-8 BOM.
        with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
            rows = (row for row in csv.reader(stream, delimiter=sep)
                    if row and (len(row) > 1 or row[0].strip()))
            fields = next(rows, [])
            missing_required = [c for c in column_set.required_columns if c not in fields]
            if missing_required:
                raise InputError(f"CSV missing required columns: {missing_required}",
                                 context={"stage": "csv_header", "path": path, "missing_columns": missing_required})

            defaults = {c: _DEFAULTS[c] for c in column_set.optional_columns if c not in fields}
            for col, value in defaults.items():
                logger.warning("Column '%s' absent; defaulting to %r", col, value)
            # Only retain schema columns, taking the first occurrence of duplicate headers.
            indices = [(col, fields.index(col)) for col in self.columns if col in fields]
            for row in islice(rows, nrows):
                if len(row) > len(fields):
                    raise InputError(f"CSV row has {len(row)} fields; expected {len(fields)}",
                                     context={"stage": "csv_record", "path": path,
                                              "record_index": len(self.records), "field_count": len(row)})
                record = {col: row[i] if i < len(row) else "" for col, i in indices}
                self.records.append({col: record[col] if col in record else defaults[col]
                                     for col in self.columns})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        return self.records[i]
