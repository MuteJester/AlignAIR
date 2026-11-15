"""Per-read metadata normalization and unique-key in-memory or disk-backed lookup."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import gzip
from itertools import chain
import json
import os
import sqlite3
import tempfile
import zlib

from ..errors import DataContractError, InputError, OperationIOError, error_context
from .airr import metadata_column
from .sequence_records import open_text

_META_DEFAULTS = ("cell_id", "barcode", "sample_id", "umi_count", "umis", "reads",
                  "duplicate_count", "consensus_count", "raw_clonotype_id", "raw_consensus_id",
                  "chain", "c_call", "is_cell", "high_confidence")
_10X_TO_AIRR = {"barcode": "cell_id", "umis": "umi_count", "reads": "consensus_count", "c_gene": "c_call"}
_SQL_PARAM_CHUNK = 900


class DuplicateMetadataId(InputError):
    """A metadata join key occurs more than once; neither row may silently win."""
    code = "ALIGNAIR_DUPLICATE_METADATA_ID"


def _duplicate(key):
    return DuplicateMetadataId(f"duplicate metadata id {key!r}; select a unique metadata join key",
                               context={"record_id": key})


def _metadata_plan(fields, id_column, keep_columns, normalize_10x, path):
    if len(set(fields)) != len(fields):
        raise InputError("metadata header columns must be unique")
    idcol = id_column if id_column is not None else next(
        (column for column in fields if column.lower() in ("sequence_id", "contig_id", "cell_id", "id", "name")), None)
    if idcol is None or idcol not in fields:
        raise InputError(f"metadata id column {id_column or '(auto)'} not found in {path} (columns: {fields})")
    keep = list(keep_columns) if keep_columns else [column for column in _META_DEFAULTS if column in fields]
    missing = [column for column in keep if column not in fields]
    if missing:
        raise InputError(f"metadata columns not in {path}: {missing} (have {fields})")
    normalization = {source: target for source, target in _10X_TO_AIRR.items() if normalize_10x and source in fields}
    columns = keep + [target for target in normalization.values() if target not in keep]
    return idcol, keep, normalization, columns


@contextmanager
def _metadata_rows(path, id_column, keep_columns, normalize_10x, rename=None):
    path = os.fspath(path)
    with error_context(stage="metadata_reading", path=path):
        try:
            with open_text(path) as stream:
                lines = enumerate(stream, 1)
                first = next((item for item in lines if item[1].strip()), (1, ""))
                number, head = first
                reader = csv.DictReader((line for _, line in chain([first], lines)),
                                        delimiter="\t" if "\t" in head else ",", strict=True)
                idcol, keep, normalization, columns = _metadata_plan(
                    reader.fieldnames or [], id_column, keep_columns, normalize_10x, path)
                renamed = [rename(column) if rename else column for column in columns]
                if (any(not isinstance(column, str) or not column for column in renamed)
                        or len(set(renamed)) != len(renamed)):
                    raise InputError("metadata output columns must be nonempty and unique after renaming")
                names = dict(zip(columns, renamed))

                def rows():
                    for raw in reader:
                        key = raw.get(idcol)
                        if not key:
                            continue
                        if any(raw.get(column) is None for column in keep):
                            raise InputError("metadata row is missing a selected field",
                                             context={"line": number + reader.line_num - 1})
                        row = {column: raw[column] for column in keep}
                        for source, target in normalization.items():
                            value = raw.get(target) or raw.get(source)
                            if value:
                                row[target] = value
                        yield key, {names[column]: value for column, value in row.items()}

                yield renamed, rows()
        except csv.Error as exc:
            raise InputError(f"malformed metadata table: {exc}",
                             context={"line": number + reader.reader.line_num - 1}) from exc
        except (UnicodeError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
            raise InputError("cannot decode metadata input") from exc
        except OSError as exc:
            raise OperationIOError("cannot read metadata input") from exc


def load_metadata(path: str | os.PathLike[str], id_column: str | None = None,
                  keep_columns=None, normalize_10x: bool = False, rename=metadata_column):
    """Return (unique-ID rows, output columns); duplicate keys are rejected in both loaders. ``rename`` names each
    output column (by default ``io.airr.metadata_column``: a column AlignAIR writes becomes ``meta_<column>``)."""
    with _metadata_rows(path, id_column, keep_columns, normalize_10x, rename) as (columns, rows):
        result = {}
        for key, row in rows:
            if key in result:
                raise _duplicate(key)
            result[key] = row
    return result, columns


class MetadataIndex:
    """Temporary SQLite lookup with atomic batch inserts and bounded lookup parameter counts.

    Close explicitly or use a context manager. Missing lookups are omitted; source IDs can repeat
    in a query, but stored keys must be unique. Failed inserts preserve all earlier committed batches.
    """

    def __init__(self):
        descriptor, self._path = tempfile.mkstemp(suffix=".alignair-meta.sqlite")
        os.close(descriptor)
        self._con = None
        try:
            self._con = sqlite3.connect(self._path)
            self._con.execute("CREATE TABLE m (id TEXT PRIMARY KEY NOT NULL, row TEXT NOT NULL)")
        except BaseException as exc:
            self.close()
            if isinstance(exc, sqlite3.Error):
                raise OperationIOError("cannot create metadata index", context={"stage": "metadata_index"}) from exc
            raise

    def _connection(self):
        if self._con is None:
            raise DataContractError("metadata index is closed")
        return self._con

    def put_many(self, items) -> None:
        connection = self._connection()
        rows = []
        for key, row in items:
            if not isinstance(key, str) or not key:
                raise DataContractError("metadata index keys must be nonempty strings")
            rows.append((key, json.dumps(row)))
        try:
            with connection:
                connection.executemany("INSERT INTO m VALUES (?, ?)", rows)
        except sqlite3.IntegrityError as exc:
            seen = set(self.get_many(key for key, _ in rows))
            for key, _ in rows:
                if key in seen:
                    raise _duplicate(key) from exc
                seen.add(key)
            raise OperationIOError("metadata index constraint failed") from exc
        except sqlite3.Error as exc:
            raise OperationIOError("cannot store metadata rows", context={"stage": "metadata_index"}) from exc

    def get_many(self, ids) -> dict:
        connection = self._connection()
        ids = list(ids)
        result = {}
        try:
            for offset in range(0, len(ids), _SQL_PARAM_CHUNK):
                batch = ids[offset:offset + _SQL_PARAM_CHUNK]
                placeholders = ",".join("?" * len(batch))
                for key, payload in connection.execute(f"SELECT id, row FROM m WHERE id IN ({placeholders})", batch):
                    result[key] = json.loads(payload)
        except sqlite3.Error as exc:
            raise OperationIOError("cannot look up metadata rows", context={"stage": "metadata_index"}) from exc
        return result

    def get(self, key, default=None):
        return self.get_many([key]).get(key, {} if default is None else default)

    def close(self) -> None:
        try:
            if self._con is not None:
                self._con.close()
                self._con = None
        finally:
            try:
                os.remove(self._path)
            except FileNotFoundError:
                pass

    def __enter__(self):
        self._connection()
        return self

    def __exit__(self, *args):
        self.close()


def build_metadata_index(path, id_column=None, keep_columns=None, normalize_10x=False, rename=metadata_column):
    """Return (owned index, output columns); close the index after joining source IDs.

    ``rename`` (as ``load_metadata``'s) is resolved once per output column. Colliding target columns fail before index
    creation. Reading and insertion use bounded batches; every failure closes and removes the index.
    """
    with _metadata_rows(path, id_column, keep_columns, normalize_10x, rename) as (columns, rows):
        index = MetadataIndex()
        try:
            batch = []
            for key, row in rows:
                batch.append((key, row))
                if len(batch) >= 5000:
                    index.put_many(batch)
                    batch = []
            if batch:
                index.put_many(batch)
        except BaseException:
            index.close()
            raise
    return index, columns


def lookup_metadata_rows(metadata, source_ids) -> list[dict] | None:
    """Align metadata to original source IDs through a mapping or batched index.

    Missing/empty source IDs receive empty rows; generated output names are never inferred as
    join keys. Duplicate source IDs share their metadata row. Indexed queries deduplicate keys.
    """
    if metadata is None:
        return None
    source_ids = tuple(source_ids)
    keys = list(dict.fromkeys(key for key in source_ids if key))
    found = metadata.get_many(keys) if hasattr(metadata, "get_many") else metadata
    return [found.get(key, {}) if key else {} for key in source_ids]
