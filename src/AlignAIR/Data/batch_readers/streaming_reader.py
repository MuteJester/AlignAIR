import csv
from .base import BatchReader

class StreamingTSVReader(BatchReader):
    def __init__(self, path, required_columns, batch_size, sep='\t'):
        self.generator = self._table_generator(path, required_columns, batch_size, sep)
        self._length = self._count_lines(path)

    def _count_lines(self, path):
        with open(path) as f:
            return sum(1 for _ in f) - 1

    def _table_generator(self, path, usecols, batch_size, sep):
        while True:
            with open(path, 'r') as file:
                reader = csv.reader(file, delimiter=sep)
                headers = next(reader)
                batch = {h: [] for h in headers}

                for row in reader:
                    for idx, val in enumerate(row):
                        col = headers[idx]
                        batch[col].append(int(val) if 'start' in col or 'end' in col else val)

                    if len(batch[headers[0]]) == batch_size:
                        yield {k: batch[k] for k in usecols}
                        batch = {h: [] for h in headers}

    def get_batch(self, _):
        return next(self.generator)

    def get_data_length(self):
        return self._length
