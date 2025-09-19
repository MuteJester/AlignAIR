import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Union
import json
from functools import reduce
from ast import literal_eval  # safer parsing for indels

from GenAIRR.dataconfig.enums import ChainType
from .SingleChainDataset import SingleChainDataset
from GenAIRR.dataconfig import DataConfig
from .MultiDataConfigContainer import MultiDataConfigContainer

from .batch_readers import StreamingTableReader
from .columnSet import ColumnSet
from .datasetBase import DatasetBase
from .encoders import AlleleEncoder, ChainTypeOneHotEncoder
from .tokenizers import CenterPaddedSequenceTokenizer


class MultiChainDataset(DatasetBase):
    """
    A dataset class for managing and aggregating data from multiple receptor chains
    (e.g., heavy and light chains, light kappa and light lambda and etc).

    This class composes multiple SingleChainDataset instances, fetches batches from
    each, and merges them into a single batch with prefixed keys. It handles
    shuffling, different dataset lengths, and heterogeneous gene presence across chains.
    This is the final, recommended implementation.

    param data_paths: List of paths to data files.
    param dataconfigs: List of DataConfig objects.
    Note: The order of data_paths and dataconfigs must correspond to each other.
    """

    def __init__(self, data_paths:List[str], dataconfigs: Union[List[DataConfig], MultiDataConfigContainer], batch_size=64, max_sequence_length=512, use_streaming=False,
                 nrows=None, seperator=',',evaluation_only=False):


        # Convert to MultiDataConfigContainer if needed
        if isinstance(dataconfigs, list):
            self.dataconfig_container = MultiDataConfigContainer(dataconfigs)
        else:
            self.dataconfig_container = dataconfigs


        self.dataconfigs = list(self.dataconfig_container)  # For backward compatibility

        self.nrows = nrows
        self.total_datasets = len(data_paths)
        self.batch_size = batch_size
        self.batch_sizes = self.dataconfig_container.equal_batch_partitioning(batch_size)


        self.max_sequence_length = max_sequence_length

        self.tokenizer = CenterPaddedSequenceTokenizer(max_length=max_sequence_length)
        self.allele_encoder = AlleleEncoder()

        self.chain_type_encoder = ChainTypeOneHotEncoder(chain_types=self.dataconfig_container.chain_types())
        #create a static batch type key we will add to each batch to indicate the chain type of each sample.
        batch_types = []
        for bsize, dcf in zip(self.batch_sizes, self.dataconfigs):
            batch_types.extend([dcf.metadata.chain_type] * bsize)
        self.batch_type_constant = self.chain_type_encoder.encode(batch_types)

        self.seperator = seperator
        self.use_streaming = use_streaming

        self.evaluation_only = evaluation_only
        if not self.evaluation_only:
            self.required_data_column_sets = [ColumnSet(has_d=dcf.metadata.has_d) for dcf in self.dataconfigs]
        else:
            self.required_data_column_sets = [['sequence'] for dcf in self.dataconfigs]


        self.batch_size = batch_size
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()
        self.data_paths = data_paths


        self.readers = {}

        for path,dcf,bsizes,req_cols in zip(data_paths, self.dataconfigs,self.batch_sizes,self.required_data_column_sets):
            self.readers[dcf.metadata.chain_type] = StreamingTableReader(
                path, req_cols, bsizes, sep=seperator, stream=use_streaming
            )

        self.data_lengths = [rd.get_data_length() for rd in self.readers.values()]
        
        # Set data_length for compatibility with parent class train generator
        # Use the minimum length to ensure all datasets can contribute to every batch
        self.data_length = min(self.data_lengths) if self.data_lengths else 0


    def derive_call_dictionaries(self):
        # Use the MultiDataConfigContainer to get combined dictionaries
        combined_dicts = self.dataconfig_container.derive_combined_allele_dictionaries()
        
        # Set attributes for backward compatibility
        for key, value in combined_dicts.items():
            setattr(self, key, value)

    def derive_call_one_hot_representation(self):
        v_alleles = list(self.v_dict)
        j_alleles = list(self.j_dict)

        self.v_allele_count = self.dataconfig_container.number_of_v_alleles
        self.j_allele_count = self.dataconfig_container.number_of_j_alleles

        self.allele_encoder.register_gene("V", v_alleles, sort=False)
        self.allele_encoder.register_gene("J", j_alleles, sort=False)

        if self.has_d:
            d_alleles = sorted(list(self.d_dict))
            if 'Short-D' not in d_alleles:
                d_alleles.append('Short-D')
            self.allele_encoder.register_gene("D", d_alleles, sort=False)
            self.d_allele_count = self.dataconfig_container.number_of_d_alleles

    @property
    def has_d(self):
        return self.dataconfig_container.has_at_least_one_d()

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

    def _has_missing_d(self):
        """
        Check if there's heterogeneous D presence across datasets.
        Returns: bool
            True if some datasets have D and others don't, False otherwise.
        """
        return self.dataconfig_container.has_missing_d()

    def _correct_for_missing_d(self, batchs):
        # if there are no missing D segments, return the batchs as is.
        if not self._has_missing_d():
            return batchs

        # at least one dataset has D, so we need to correct the batches and add all the d labels to the batches that do not have D.
        for batch, dcf in zip(batchs, self.dataconfigs):
            if not dcf.metadata.has_d:
                # Use the sequence length as the reference for batch size
                batch_size = len(batch['sequence'])  # ← Use sequence as the reliable reference
                batch['d_call'] = ['Short-D'] * batch_size
                batch['d_sequence_start'] = batch['v_sequence_end']
                batch['d_sequence_end'] = batch['v_sequence_end']
        return batchs

    def _seperate_alleles(self,allele_list):
        """
        Converts a comma-separated strings of alleles into a set of unique alleles.
        Args:
            allele_list (str): Comma-separated string of alleles.
        Returns:
            set: A set of unique alleles.
        """
        if isinstance(allele_list, str):
            return set(allele_list.split(','))
        elif isinstance(allele_list, str):
            return set(allele_list.split(','))
        else:
            raise ValueError(f"Expected str or list, got {type(allele_list)}")


    def _get_single_batch(self, pointer):
        # Read data from each reader
        batchs = []
        for (chain_type, reader) in self.readers.items():
            data = reader.get_batch(pointer)
            batchs.append(data)
        # correct for missing D segments
        batchs = self._correct_for_missing_d(batchs)
        # Concatenate all batches into a single dictionary
        batch = reduce(lambda acc, d: {k: acc[k] + d[k] for k in acc}, batchs)
        # convert all the lists to numpy arrays
        for k in batch:
            if isinstance(batch[k], list):
                batch[k] = np.array(batch[k])

        # add type key for each chain type in the batch
        batch['chain_type'] = self.batch_type_constant

        assert len(batch['sequence']) == self.batch_size, f"Batch size mismatch: {len(batch['sequence'])} != {self.batch_size}"



        # Encoded sequence in batch and collect the padding sizes applied to each sequences
        encoded_sequences,paddings = self.tokenizer.encode_and_pad_center(batch['sequence'])

        # use the padding sizes collected to adjust the start/end positions of the alleles
        targets = [f'{gene}_sequence' for gene in self._loaded_genes]
        for _gene in targets:
            for _position in ['start', 'end']:
                batch[_gene + '_' + _position] += paddings

        x = {"tokenized_sequence": encoded_sequences}

        indel_counts = []
        for indels in batch['indels']:
            if isinstance(indels, (dict, list, tuple)):
                indel_counts.append(len(indels))
            else:
                try:
                    parsed = literal_eval(indels) if isinstance(indels, str) else {}
                    indel_counts.append(len(parsed) if isinstance(parsed, (dict, list, tuple)) else 0)
                except Exception:
                    indel_counts.append(0)

        # Convert Comma Seperated Allele Ground Truth Labels into Lists
        v_alleles = list(map(self._seperate_alleles, batch['v_call']))
        j_alleles = list(map(self._seperate_alleles, batch['j_call']))

        def to_float_bool(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            if isinstance(x, (int, float)):
                return 1.0 if x != 0 else 0.0
            s = str(x).strip().lower()
            if s in {'true','1','yes','y','t'}:
                return 1.0
            if s in {'false','0','no','n','f'}:
                return 0.0
            return 0.0
        y = {
            "v_start": batch['v_sequence_start'].reshape(-1, 1),
            "v_end": batch['v_sequence_end'].reshape(-1, 1),
            "j_start": batch['j_sequence_start'].reshape(-1, 1),
            "j_end": batch['j_sequence_end'].reshape(-1, 1),
            "v_allele": self.one_hot_encode_allele("V", v_alleles),
            "j_allele": self.one_hot_encode_allele("J", j_alleles),
            'mutation_rate': batch['mutation_rate'].reshape(-1, 1),
            'indel_count': np.array(indel_counts).reshape(-1, 1),
            'productive': np.array([to_float_bool(i) for i in batch['productive']]).reshape(-1, 1),
            'chain_type': batch['chain_type']
        }
        if self.has_d:
            d_alleles = list(map(self._seperate_alleles, batch['d_call']))
            y["d_allele"] = self.one_hot_encode_allele("D", d_alleles)
            y['d_start'] = batch['d_sequence_start'].reshape(-1, 1)
            y['d_end'] = batch['d_sequence_end'].reshape(-1, 1)


        # shuffle the order of the values in x and y to ensure the batches are shuffled
        num_samples = x['tokenized_sequence'].shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Shuffle x and y dictionaries using the same shuffled indices
        for key in x:
            x[key] = x[key][indices]
        for key in y:
            y[key] = y[key][indices]

        return x, y

    def _get_tf_dataset_params(self):

        x, y = self._get_single_batch(self.batch_size)

        output_types = ({k: tf.float32 for k in x}, {k: tf.float32 for k in y})

        output_shapes = ({k: (self.batch_size,) + x[k].shape[1:] for k in x},
                         {k: (self.batch_size,) + y[k].shape[1:] for k in y})

        return output_types, output_shapes

    def _train_generator(self):
        pointer = 0
        batch_count = 0
        while True:
            pointer += self.batch_size
            if pointer >= self.data_length:
                pointer = self.batch_size

            batch_x, batch_y = self._get_single_batch(pointer)
            batch_count += 1
            yield batch_x, batch_y

    def get_train_dataset(self):
        output_types, output_shapes = self._get_tf_dataset_params()

        dataset = tf.data.Dataset.from_generator(
            lambda: self._train_generator(),
            output_types=output_types,
            output_shapes=output_shapes,
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def generate_model_params(self):
        params = {
            "max_seq_length": self.max_sequence_length,
            'dataconfigs':self.dataconfig_container}
        return params

    def __len__(self):
        return self.data_length

