import numpy as np
import pandas as pd
import csv
from GenAIRR.dataconfig import DataConfig
import tensorflow as tf
from docutils.nodes import target
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

from .columnSet import ColumnSet
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

        self.v_dict = None
        self.d_dict = None
        self.j_dict = None

        self.max_sequence_length = max_sequence_length

        self.dataconfig = dataconfig
        self.tokenizer = CenterPaddedSequenceTokenizer(max_length=max_sequence_length)
        self.allele_encoder = AlleleEncoder()

        self.seperator = seperator
        self.use_streaming  = use_streaming
        self.required_data_columns = ColumnSet() if required_data_columns is None else required_data_columns
        self.batch_size = batch_size
        self.add_allele_dictionaries()
        self.register_alleles_to_ohe()
        self.data_path = data_path

        if use_streaming:
            self.reader = StreamingTSVReader(data_path, self.required_data_columns, batch_size, sep=seperator)
        else:
            self.reader = PandasBatchReader(data_path, self.required_data_columns, batch_size, nrows=nrows,
                                            sep=seperator)

        self.data_length = self.reader.get_data_length()

    def register_alleles_to_ohe(self):
        """
        Register alleles to the one-hot encoder based on the derived dictionaries.
        Returns:

        """
        v_alleles = sorted(list(self.v_dict))
        j_alleles = sorted(list(self.j_dict))

        self.v_allele_count = len(v_alleles)
        self.j_allele_count = len(j_alleles)

        self.allele_encoder.register_gene("V", v_alleles, sort=False)
        self.allele_encoder.register_gene("J", j_alleles, sort=False)

        if self.has_d:
            d_alleles = sorted(list(self.d_dict)) + ['Short-D']
            self.d_allele_count = len(d_alleles)
            self.allele_encoder.register_gene("D", d_alleles, sort=False)


    @property
    def has_d(self):
        return self.dataconfig.metadata.has_d

    def add_allele_dictionaries(self):
        self.v_dict = {i.name:i for i in self.dataconfig.allele_list('v')}
        self.j_dict = {i.name:i for i in self.dataconfig.allele_list('j')}

        if self.has_d:
            self.d_dict = {i.name:i for i in self.dataconfig.allele_list('d')}

    def encode_and_equal_pad_sequence(self, sequence):
        return self.tokenizer.encode_and_pad_center(sequence)

    def get_ohe_reverse_mapping(self):
        return self.allele_encoder.get_reverse_mapping()

    def one_hot_encode_allele(self, gene, ground_truth_labels):
        return self.allele_encoder.encode(gene, ground_truth_labels)

    @property
    def _loaded_genes(self):
        """
        return the list of loaded genes in the dataset with the context of the dataconfig.
        Returns:
            List[str]: List of gene names that are loaded in the dataset.
        """
        genes = ['v', 'j']
        if self.has_d:
            genes.append('d')
        return genes



    def _get_single_batch(self, pointer):
        # Read Batch from Dataset
        batch = self.reader.get_batch(pointer)
        batch = pd.DataFrame(batch)

        # Encoded sequence in batch and collect the padding sizes applied to each sequences
        encoded_sequences, paddings = self.tokenizer.encode_and_pad_center(batch['sequence'])
        # use the padding sizes collected to adjust the start/end positions of the alleles
        targets = [f'{gene}_sequence' for gene in self._loaded_genes]
        for _gene in targets:
            for _position in ['start', 'end']:
                batch.loc[:, _gene + '_' + _position] += paddings

        x = {"tokenized_sequence": encoded_sequences}

        indel_counts = []
        for ax, row in batch.iterrows():
            indels = eval(row['indels'])
            indel_counts.append(len(indels))

        # Convert Comma Seperated Allele Ground Truth Labels into Lists
        v_alleles = batch.v_call.apply(lambda x: set(x.split(',')))
        j_alleles = batch.j_call.apply(lambda x: set(x.split(',')))

        bool_cast = lambda x: 1. if any([x=='True', x ==True]) else 0.
        y = {
            "v_start": batch.v_sequence_start.values.reshape(-1, 1),
            "v_end": batch.v_sequence_end.values.reshape(-1, 1),
            "j_start": batch.j_sequence_start.values.reshape(-1, 1),
            "j_end": batch.j_sequence_end.values.reshape(-1, 1),
            "v_allele": self.one_hot_encode_allele("V", v_alleles),
            "j_allele": self.one_hot_encode_allele("J", j_alleles),
            'mutation_rate': batch.mutation_rate.values.reshape(-1, 1),
            'indel_count': np.array(indel_counts).reshape(-1, 1),
            'productive': np.array([bool_cast(i) for i in batch.productive]).reshape(-1, 1)

        }
        if self.has_d:
            d_alleles = batch.d_call.apply(lambda x: set(x.split(',')))
            y["d_allele"] = self.one_hot_encode_allele("D", d_alleles)
            y['d_start'] = batch.d_sequence_start.values.reshape(-1, 1)
            y['d_end'] = batch.d_sequence_end.values.reshape(-1, 1)

        return x, y

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

    def generate_model_params(self):
        params =  {
            "max_seq_length": self.max_sequence_length,
            'dataconfig':self.dataconfig
        }
        return params

    def __len__(self):
        return self.data_length

