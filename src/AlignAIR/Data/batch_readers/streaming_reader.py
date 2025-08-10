import csv
import warnings

from .base import BatchReader

class StreamingTSVReader(BatchReader):
    def __init__(self, path, required_columns, batch_size, sep='\t'):
        self.generator = self._table_generator(path, required_columns, batch_size, sep)
        self._length = self._count_lines(path)

    def _count_lines(self, path):
        with open(path) as f:
            return sum(1 for _ in f) - 1

    # def _table_generator(self, path, usecols, batch_size, sep):
    #     while True:
    #         with open(path, 'r') as file:
    #             reader = csv.reader(file, delimiter=sep)
    #             try:
    #                 headers = next(reader)
    #             except StopIteration:
    #                 # This handles the case where the file is empty
    #                 break  # Exit the while loop if file is empty
    #
    #             batch = {h: [] for h in headers}
    #             # We add line_num to provide more informative warnings
    #             for line_num, row in enumerate(reader, start=2):  # Start from line 2
    #                 try:
    #                     processed_row_data = {}
    #                     for idx, val in enumerate(row):
    #                         col = headers[idx]
    #                         # The conversion logic that might fail is now inside the try block
    #                         processed_row_data[col] = int(val) if 'start' in col or 'end' in col else val
    #
    #                     for col, value in processed_row_data.items():
    #                         batch[col].append(value)
    #
    #                 except ValueError:
    #                     warnings.warn(
    #                         f"Skipping row {line_num} due to a data conversion error. Row content: {row}"
    #                     )
    #                     continue  #skips adding the faulty row to the batch.
    #
    #
    #                 # Check if the batch is full and yield it
    #                 if len(batch[headers[0]]) == batch_size:
    #                     yield {k: batch[k] for k in usecols if k in batch}
    #                     batch = {h: [] for h in headers}

    def _table_generator(self, path, usecols, batch_size, sep):
        """
        Generates batches of data from a CSV, skipping rows with conversion errors.
        """
        while True:
            try:
                with open(path, 'r') as file:
                    reader = csv.reader(file, delimiter=sep)

                    # Safely get headers and handle empty files
                    try:
                        headers = next(reader)
                    except StopIteration:
                        return  # Stop generation if file is empty

                    batch = {h: [] for h in headers}

                    # Use enumerate for accurate line numbers in warnings
                    for line_num, row in enumerate(reader, start=2):
                        try:
                            # 1. Process the entire row's values into a temporary list first.
                            processed_values = []
                            for idx, val in enumerate(row):
                                col = headers[idx]
                                # This is the line that can fail
                                converted_val = int(val) if 'start' in col or 'end' in col else val
                                processed_values.append(converted_val)

                            # 2. If the above loop succeeds, add the row to the batch.
                            # This ensures the batch isn't partially updated.
                            for idx, final_val in enumerate(processed_values):
                                batch[headers[idx]].append(final_val)

                        except ValueError:
                            # 3. If int() fails, warn the user and skip this row.
                            warnings.warn(
                                f"⚠️ Skipping row {line_num} due to a conversion error. Row content: {row}"
                            )
                            continue  # Move to the next row in the file

                        # Yield a batch when it's full
                        if headers and len(batch[headers[0]]) == batch_size:
                            yield {k: batch[k] for k in usecols if k in batch}
                            batch = {h: [] for h in headers}


            except FileNotFoundError:
                warnings.warn(f"File not found at path: {path}")
                return  # Stop the generator if the file doesn't exist

    def get_batch(self, _):
        return next(self.generator)

    def get_data_length(self):
        return self._length
