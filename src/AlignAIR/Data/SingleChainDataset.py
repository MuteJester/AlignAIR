# In your SingleChainDataset.py file

import numpy as np
import pandas as pd

from .columnSet import ColumnSet
from ..Data.datasetBase import DatasetBase
from GenAIRR.dataconfig import DataConfig


class SingleChainDataset(DatasetBase):
    """
    A dataset class for a single chain (e.g., heavy or light).
    """

    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=576, use_streaming=False,
                 nrows=None, seperator=',',evaluation_only=False):
        if evaluation_only:
            self.required_data_columns = ['sequence'] # we only need the sequence for evaluation
        else:
            self.required_data_columns = ColumnSet(has_d=dataconfig.metadata.has_d)

        super().__init__(data_path, dataconfig, batch_size, max_sequence_length, use_streaming, nrows, seperator,
                         required_data_columns=self.required_data_columns)