import csv
import logging
from ast import literal_eval
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional

from .base import BatchReader

Converter = Callable[[Any], Any]


class StreamingTableReader(BatchReader):
    def __init__(
        self,
        path: str,
        required_columns: Iterable[str],
        batch_size: int,
        sep: str = '\t',
        converters: Optional[Dict[str, Converter]] = None,
        stream: bool = True,
        error_policy: str = 'skip',  # 'skip' | 'raise' | 'coerce'
        coerce_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Streaming reader that yields batches containing only required columns.

        Args:
            path: Path to delimited file.
            required_columns: Iterable of required column names (e.g., ColumnSet).
            batch_size: Number of rows per yielded batch.
            sep: Field delimiter.
            converters: Optional dict[col_name -> callable(str)->Any] to convert values.
                        Defaults include:
                          - '*_start'/'*_end' -> int
                          - 'mutation_rate'   -> float
                          - 'productive'      -> float 0.0/1.0
                          - 'indels'          -> dict (literal_eval if string)
        """
        self.sep = sep
        self.required_columns: List[str] = list(required_columns)
        self.batch_size = batch_size
        self.stream = stream
        self.converters = converters
        self.error_policy = error_policy
        self.coerce_defaults = coerce_defaults or {}
        self._logger = logging.getLogger(__name__)

        self.generator = None
        self._in_memory = None
        if self.stream:
            self.generator = self._table_generator(path, self.required_columns, batch_size, sep, converters)
            self._length = self._count_lines(path)
        else:
            # Load the entire table once into memory (only required columns, with conversions)
            self._in_memory = self._read_all(path, self.required_columns, sep, converters)
            # data length is number of rows
            self._length = len(next(iter(self._in_memory.values()))) if self._in_memory else 0

    def _count_lines(self, path: str) -> int:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            return sum(1 for _ in f) - 1

    # ---- Small helpers -----------------------------------------------------------
    def _open_csv(self, path: str, sep: str):
        return csv.reader(open(path, 'r', encoding='utf-8', newline=''), delimiter=sep)

    def _read_headers(self, reader) -> List[str]:
        try:
            return next(reader)
        except StopIteration:
            return []

    def _prepare_indices(self, headers: List[str], required_cols: List[str]) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        header_to_idx = {h: i for i, h in enumerate(headers)}
        missing = sorted([c for c in required_cols if c not in header_to_idx])
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Present headers: {headers}"
            )
        req_indices = [(col, header_to_idx[col]) for col in required_cols]
        return req_indices, header_to_idx

    def _build_converters(self, user: Optional[Dict[str, Converter]]) -> Dict[str, Converter]:
        def _to_int(x: Any) -> int:
            if x is None or x == '':
                raise ValueError('Missing required numeric value')
            return int(x)

        def _to_float(x: Any) -> float:
            if x is None or x == '':
                raise ValueError('Missing required float value')
            return float(x)

        def _to_float_bool(x: Any) -> float:
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            if x is None:
                raise ValueError('Missing required boolean value')
            s = str(x).strip().lower()
            if s in {'true', '1', 'yes', 'y', 't'}:
                return 1.0
            if s in {'false', '0', 'no', 'n', 'f'}:
                return 0.0
            raise ValueError(f'Invalid boolean value: {x}')

        def _to_indels(x: Any):
            if isinstance(x, (dict, list, tuple)):
                return x
            if x is None:
                return {}
            s = str(x).strip()
            if s == '':
                return {}
            try:
                parsed = literal_eval(s)
                return parsed if isinstance(parsed, (dict, list, tuple)) else {}
            except Exception:
                self._logger.warning(f"Failed to parse indels value; defaulting to empty. Value: {x}")
                return {}

        defaults: Dict[str, Converter] = {
            'mutation_rate': _to_float,
            'productive': _to_float_bool,
            'indels': _to_indels,
        }
        merged = dict(defaults)
        if user:
            merged.update(user)
        # Special rule for any required *_start/_end columns
        merged['__start_end__'] = _to_int
        return merged

    def _convert_value(self, col: str, val: Any, converters: Dict[str, Converter]) -> Any:
        if isinstance(val, str):
            val = val.strip()
        conv = converters.get(col)
        if conv is None and (col.endswith('_start') or col.endswith('_end')):
            conv = converters.get('__start_end__')
        return conv(val) if conv else val

    def _default_for(self, col: str) -> Any:
        if col in self.coerce_defaults:
            return self.coerce_defaults[col]
        if col.endswith('_start') or col.endswith('_end'):
            return 0
        if col == 'mutation_rate':
            return 0.0
        if col == 'productive':
            return 0.0
        if col == 'indels':
            return {}
        return ''

    def _process_row(self, row: List[str], req_indices: List[Tuple[str, int]], converters: Dict[str, Converter]) -> Dict[str, Any]:
        processed: Dict[str, Any] = {}
        for col, idx in req_indices:
            try:
                val = row[idx]
            except IndexError:
                val = ''
            try:
                processed[col] = self._convert_value(col, val, converters)
            except ValueError:
                if self.error_policy == 'coerce':
                    processed[col] = self._default_for(col)
                else:
                    raise
        return processed

    def _table_generator(self, path: str, usecols: Iterable[str], batch_size: int, sep: str, converters: Optional[Dict[str, Converter]]):
        """
        Generate batches from a delimited file, only reading required columns.

        Improvements:
        - Precompute indices of required columns from headers.
        - Build batch dict only for required columns.
        - Only convert types for required columns via per-column converters.
        - Skip a row only if conversion for a required column fails.
        """
        # Normalize usecols to a list (ColumnSet is iterable)
        required_cols = list(usecols)
        conv = self._build_converters(converters)

        while True:
            try:
                reader = self._open_csv(path, sep)
                headers = self._read_headers(reader)
                if not headers:
                    return
                req_indices, _ = self._prepare_indices(headers, required_cols)

                # Initialize batch storage only for required columns
                batch = {col: [] for col in required_cols}

                # Use enumerate for accurate line numbers in warnings
                for line_num, row in enumerate(reader, start=2):
                    try:
                        # Collect processed values for required columns only
                        processed_row = self._process_row(row, req_indices, conv)

                        # If all conversions succeeded, append to batch
                        for col in required_cols:
                            batch[col].append(processed_row[col])

                    except ValueError:
                        # Conversion failure for a required column
                        if self.error_policy == 'skip':
                            self._logger.warning(
                                f"Skipping row {line_num} due to a conversion error in required columns. Row content: {row}"
                            )
                            continue
                        else:
                            raise

                    # Yield a batch when it's full
                    first_key = required_cols[0] if required_cols else None
                    if first_key and len(batch[first_key]) == batch_size:
                        yield batch
                        batch = {col: [] for col in required_cols}


            except FileNotFoundError:
                self._logger.warning(f"File not found at path: {path}")
                return  # Stop the generator if the file doesn't exist

    def _read_all(self, path: str, required_cols: List[str], sep: str, converters: Optional[Dict[str, Converter]]):
        """Read the entire table into memory applying the same converter logic."""
        data: Dict[str, List[Any]] = {col: [] for col in required_cols}
        try:
            reader = self._open_csv(path, sep)
            headers = self._read_headers(reader)
            if not headers:
                return data
            req_indices, _ = self._prepare_indices(headers, required_cols)
            conv = self._build_converters(converters)
            for line_num, row in enumerate(reader, start=2):
                try:
                    processed_row = self._process_row(row, req_indices, conv)
                    for col in required_cols:
                        data[col].append(processed_row[col])
                except ValueError:
                    if self.error_policy == 'skip':
                        self._logger.warning(
                            f"Skipping row {line_num} due to a conversion error in required columns. Row content: {row}"
                        )
                        continue
                    else:
                        raise
        except FileNotFoundError:
            self._logger.warning(f"File not found at path: {path}")
            return data
        return data

    def get_batch(self, pointer: int):
        if self.stream:
            if self.generator is None:
                raise RuntimeError('Streaming generator not initialized')
            return next(self.generator)
        else:
            # Non-streaming mode: slice a batch based on the pointer
            data = self._in_memory or {}
            length = self._length
            start = (pointer or 0) % max(length, 1)
            end = start + self.batch_size
            result: Dict[str, List[Any]] = {}
            for k, col in data.items():
                if length == 0:
                    result[k] = []
                    continue
                if end <= length:
                    result[k] = col[start:end]
                else:
                    part1 = col[start:]
                    part2 = col[: (end % length)]
                    result[k] = part1 + part2
            return result

    def get_all(self) -> Dict[str, List[Any]]:
        """Return the entire table (non-stream mode only)."""
        if self.stream:
            raise RuntimeError('get_all() is only available in non-stream mode')
        return self._in_memory or {}

    def get_data_length(self):
        return self._length
