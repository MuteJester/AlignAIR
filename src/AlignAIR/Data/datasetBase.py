import numpy as np
import pandas as pd
import csv
from GenAIRR.utilities import DataConfig
import tensorflow as tf
from tqdm.auto import tqdm
from abc import ABC, abstractmethod


class DatasetBase(ABC):

    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=512, batch_read_file=False,
                 nrows=None, seperator=','):
        self.max_sequence_length = max_sequence_length

        self.dataconfig = dataconfig
        self.max_seq_length = max_sequence_length
        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
        self.seperator = seperator
        self.batch_read_file = batch_read_file
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_allele',
                                      'd_allele', 'j_allele', 'mutation_rate']
        self.batch_size = batch_size
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()
        self.data_path = data_path

        if not self.batch_read_file:
            self.data = pd.read_table(data_path, usecols=self.required_data_columns, nrows=nrows, sep=self.seperator)
            self.data_length = len(self.data)
        else:
            self.get_data_batch_generator(data_path)
            self.data_length = self.count_tsv_size(data_path)

    def get_data_batch_generator(self, path_to_data):
        self.data = self.table_generator(path_to_data, batch_size=self.batch_size, usecols=self.required_data_columns,
                                         seperator=self.seperator)

    @abstractmethod
    def derive_call_one_hot_representation(self):
        pass

    @abstractmethod
    def derive_call_dictionaries(self):
        pass

    def generate_batch(self, pointer):
        if not self.batch_read_file:
            data = {
                col: [] for col in self.required_data_columns
            }
            for idx, row in self.data.iloc[(pointer - self.batch_size):pointer, :].iterrows():
                for col in self.required_data_columns:
                    data[col].append(row[col])
            return data
        else:
            read_batch = next(self.data)
            return read_batch

    def encode_and_equal_pad_sequence(self, sequence):
        """Encodes a sequence of nucleotides and pads it to the specified maximum length, equally from both sides.

        Args:
            sequence: A sequence of nucleotides.

        Returns:
            A padded sequence, and the start and end indices of the unpadded sequence.
        """

        encoded_sequence = np.array([self.tokenizer_dictionary[i] for i in sequence])
        padding_length = self.max_seq_length - len(encoded_sequence)
        iseven = padding_length % 2 == 0
        pad_size = padding_length // 2
        if iseven:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size), 'constant', constant_values=(0, 0))
        else:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size + 1), 'constant', constant_values=(0, 0))
        return encoded_sequence, pad_size

    def encode_and_pad_sequence_end(self, sequence):
        """Encodes a sequence of nucleotides and pads it to the specified maximum length, only at the end.

        Args:
            sequence: A sequence of nucleotides.

        Returns:
            A padded sequence, and the start index of the unpadded sequence.
        """

        encoded_sequence = np.array([self.tokenizer_dictionary[i] for i in sequence])
        padding_length = self.max_seq_length - len(encoded_sequence)

        # Pad only at the end
        encoded_sequence = np.pad(encoded_sequence, (0, padding_length), 'constant', constant_values=(0, 0))

        # The start index of the unpadded sequence is always 0 since we're not padding at the start
        return encoded_sequence, 0

    @abstractmethod
    def get_ohe_reverse_mapping(self):
        pass

    def one_hot_encode_allele(self, gene, ground_truth_labels):
        allele_count = self.properties_map[gene]['allele_count']
        allele_call_ohe = self.properties_map[gene]['allele_call_ohe']
        result = []
        for sample in ground_truth_labels:
            ohe = np.zeros(allele_count)
            for allele in sample:
                if allele in allele_call_ohe:
                    ohe[allele_call_ohe[allele]] = 1
            result.append(ohe)
        return np.vstack(result)

    def encode_and_pad_sequences(self, sequences):
        padded_arrays = []
        paddings = []
        for sequence in sequences:
            pad_seq, pad_size = self.encode_and_equal_pad_sequence(sequence)
            padded_arrays.append(pad_seq)
            paddings.append(pad_size)
        return np.vstack(padded_arrays), np.array(paddings)

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
        self.get_data_batch_generator(self.data_path)
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

    def tokenize_sequences(self, sequences, verbose=False):
        padded_sequences = []

        if verbose:
            iterator = tqdm(sequences, total=len(sequences))
        else:
            iterator = sequences

        for seq in iterator:
            padded_array, padding_size = self.encode_and_equal_pad_sequence(seq)
            padded_sequences.append(padded_array)
        padded_sequences = np.vstack(padded_sequences)

        return padded_sequences

    def __len__(self):
        return len(self.data)

    @staticmethod
    def count_tsv_size(path):
        with open(path) as fp:
            for (count, _) in enumerate(fp, 1):
                pass
            return count - 1

    @staticmethod
    def table_generator(path, batch_size, usecols=None, loop=True, seperator='\t'):
        with open(path, 'r') as file:
            while True:
                reader = csv.reader(file, delimiter=seperator)
                headers = next(reader)  # Skip header row
                if len(set(usecols) & set(headers)) < len(usecols):
                    for col in usecols:
                        if col not in headers:
                            print('!!!!!!!!!!!!!!!!!!!!!!!!!!', col, '!!!!!!!!!!!!!!!')
                    raise ValueError(f"Not all required columns were provided in train data file: {path}")

                batch = {i: [] for i in headers}
                for row in reader:
                    for en, value in enumerate(row):
                        if any([kw in headers[en] for kw in ['start', 'end']]):
                            batch[headers[en]].append(int(value))
                        else:
                            batch[headers[en]].append(value)

                    if len(batch[headers[0]]) == batch_size:
                        yield batch
                        batch = {i: [] for i in headers}
                file.seek(0)  # Reset file pointer to the beginning
