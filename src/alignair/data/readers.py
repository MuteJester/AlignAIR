"""CSV/TSV reader producing per-row dicts; defaults missing optional columns."""
import logging
import pandas as pd

from .column_schema import ColumnSet

logger = logging.getLogger(__name__)

_DEFAULTS = {"productive": 1.0, "indels": ""}


class CsvTableReader:
    def __init__(self, path: str, column_set: ColumnSet, sep: str = ",",
                 nrows: int | None = None):
        df = pd.read_csv(path, sep=sep, nrows=nrows, dtype=str, keep_default_na=False)

        missing_required = [c for c in column_set.required_columns if c not in df.columns]
        if missing_required:
            raise ValueError(f"CSV missing required columns: {missing_required}")

        for col in column_set.optional_columns:
            if col not in df.columns:
                df[col] = _DEFAULTS[col]
                logger.warning("Column '%s' absent; defaulting to %r", col, _DEFAULTS[col])

        self.columns = column_set.as_list()
        self.records = df[self.columns].to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        return self.records[i]
