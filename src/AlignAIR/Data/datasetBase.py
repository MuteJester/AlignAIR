import numpy as np
import pandas as pd
import csv
from GenAIRR.dataconfig import DataConfig
import tensorflow as tf
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from .tokenizers import CenterPaddedSequenceTokenizer
from .encoders import AlleleEncoder
from .batch_readers import PandasBatchReader, StreamingTSVReader

class DatasetBase(ABC):
    """
    Base dataset class for AIRR modeling.

    Provides infrastructure for tokenizing sequences, encoding alleles, and reading data
    in either full or streaming mode. Subclasses must implement dictionary derivation
    and batch assembly logic.
    """
    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=512, use_streaming=False,
                 nrows=None, seperator=',',required_data_columns=None):
        self.max_sequence_length = max_sequence_length

        self.dataconfig = dataconfig
        self.tokenizer = CenterPaddedSequenceTokenizer(max_length=max_sequence_length)
        self.allele_encoder = AlleleEncoder()

        self.seperator = seperator
        self.use_streaming  = use_streaming
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_call',
                                      'd_call', 'j_call', 'mutation_rate', 'indels', 'productive'] if required_data_columns is None else required_data_columns
        self.batch_size = batch_size
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()
        self.data_path = data_path

        if use_streaming:
            self.reader = StreamingTSVReader(data_path, self.required_data_columns, batch_size, sep=seperator)
        else:
            self.reader = PandasBatchReader(data_path, self.required_data_columns, batch_size, nrows=nrows,
                                            sep=seperator)

        self.data_length = self.reader.get_data_length()


    @abstractmethod
    def derive_call_one_hot_representation(self):
        pass

    @abstractmethod
    def derive_call_dictionaries(self):
        pass

    def encode_and_equal_pad_sequence(self, sequence):
        return self.tokenizer.encode_and_pad_center(sequence)

    def get_ohe_reverse_mapping(self):
        return self.allele_encoder.get_reverse_mapping()

    def one_hot_encode_allele(self, gene, ground_truth_labels):
        return self.allele_encoder.encode(gene, ground_truth_labels)

    @abstractmethod
    def _get_single_batch(self, pointer):
        pass

    def _get_tf_dataset_params(self):

        x, y = self._get_single_batch(self.batch_size)

        output_types = ({k: tf.float32 for k in x}, {k: tf.float32 for k in y})

        output_shapes = ({k: (self.batch_size,) + x[k].shape[1:] for k in x},
                         {k: (self.batch_size,) + y[k].shape[1:] for k in y})

        return output_types, output_shapes

    def _train_generator(self):
        pointer = 0
        while True:
            pointer += self.batch_size
            if pointer >= self.data_length:
                pointer = self.batch_size

            batch_x, batch_y = self._get_single_batch(pointer)

            if len(batch_x['tokenized_sequence']) != self.batch_size:
                pointer = 0
                continue
            else:
                yield batch_x, batch_y

    def get_train_dataset(self):
        output_types, output_shapes = self._get_tf_dataset_params()

        dataset = tf.data.Dataset.from_generator(
            lambda: self._train_generator(),
            output_types=output_types,
            output_shapes=output_shapes,
        )

        return dataset

    @abstractmethod
    def generate_model_params(self):
        pass

    def __len__(self):
        return self.data_length

