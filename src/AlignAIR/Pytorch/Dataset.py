import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import csv
from tqdm.auto import tqdm
import csv
import torch
from torch.utils.data import Dataset


class CSVReaderDataset(Dataset):
    def __init__(self, csv_file, preprocessor=None, batch_size=64, separator=','):
        """
        A PyTorch Dataset that reads from a CSV file efficiently in batches and caches the rows.

        Args:
            csv_file (str): Path to the CSV file.
            preprocessor (callable, optional): A function/transform to apply to the data.
            batch_size (int): Number of rows to read at a time.
            separator (str): Delimiter for the CSV file.
        """
        self.csv_file = csv_file
        self.transform = preprocessor
        self.batch_size = batch_size
        self.separator = separator

        # Count the total number of rows in the CSV file (excluding the header)
        self.total_rows = self._count_csv_rows()

        # Open the file and prepare a reader
        self.file = open(self.csv_file, 'r')
        self.reader = csv.reader(self.file, delimiter=self.separator)
        self.header = next(self.reader)  # Skip header
        self.skip_index = False
        # remove index column
        if '' in self.header:
            self.header.remove('')
            self.skip_index = True

        # Initialize an internal cache
        self.cache = []
        self.current_index = 0

    def _count_csv_rows(self):
        """Counts the total number of rows in the CSV file, excluding the header."""
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file, delimiter=self.separator)
            next(reader)  # Skip header
            return sum(1 for _ in reader)

    def __len__(self):
        """Returns the total number of rows in the dataset."""
        return self.total_rows

    def _fill_cache(self):
        """Reads the next batch of rows from the CSV file into the cache."""
        self.cache = []
        for _ in range(self.batch_size):
            try:
                row = next(self.reader)
                self.cache.append({n:v for n,v in zip(self.header, row[self.skip_index:])})
            except StopIteration:
                # If the end of the file is reached, reset the reader
                self._reset_reader()
                break

    def _reset_reader(self):
        """Resets the file reader to the beginning of the file."""
        self.file.seek(0)
        self.skip_index = False
        self.reader = csv.reader(self.file, delimiter=self.separator)
        self.header = next(self.reader)  # Skip header
        # remove index column
        if '' in self.header:
            self.header.remove('')
            self.skip_index = True

    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the data point.

        Returns:
            processed_row: The processed row of data.
        """
        # If the cache is empty, refill it
        if len(self.cache) == 0:
            self._fill_cache()

        # Retrieve the row at the current index
        raw_row = self.cache[self.current_index]

        # Move the pointer forward
        self.current_index += 1

        # If the pointer reaches the cache limit, refill the cache
        if self.current_index >= len(self.cache):
            self._fill_cache()
            self.current_index = 0

        # Apply any optional preprocessing
        if self.transform:
            return self.transform(raw_row)

        return raw_row

    def __del__(self):
        """Ensure the file is closed when the dataset is deleted."""
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()



