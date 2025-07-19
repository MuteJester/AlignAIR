import pandas as pd
from .base import BatchReader

class PandasBatchReader(BatchReader):
    def __init__(self, path, required_columns, batch_size, nrows=None, sep=','):
        self.df = pd.read_table(path, usecols=required_columns, nrows=nrows, sep=sep)
        self.required_columns = required_columns
        self.batch_size = batch_size
        self.df_length = len(self.df)

    def get_batch(self, pointer):
        pointer = pointer % self.df_length
        end = pointer + self.batch_size

        if end <= self.df_length:
            batch_df = self.df.iloc[pointer:end]
        else:
            # Wrap around to the beginning
            part1 = self.df.iloc[pointer:]
            part2 = self.df.iloc[:end % self.df_length]
            batch_df = pd.concat([part1, part2], axis=0)

        return batch_df.to_dict(orient='list')

    def get_data_length(self):
        return self.df_length

    def __len__(self):
        return self.df_length
