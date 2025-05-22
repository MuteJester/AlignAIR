import pandas as pd
from .base import BatchReader


class PandasBatchReader(BatchReader):
    def __init__(self, path, required_columns, batch_size, nrows=None, sep=','):
        self.df = pd.read_table(path, usecols=required_columns, nrows=nrows, sep=sep)
        self.required_columns = required_columns
        self.batch_size = batch_size

    def get_batch(self, pointer):
        return self.df.iloc[(pointer - self.batch_size):pointer].to_dict(orient='list')

    def get_data_length(self):
        return len(self.df)
