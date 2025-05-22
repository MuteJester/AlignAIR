import numpy as np
import pandas as pd

from ..Data.datasetBase import DatasetBase
from GenAIRR.dataconfig import DataConfig


class HeavyChainDataset(DatasetBase):
    """
    A dataset class for mounting heavy chain repertoire data.
    Attributes:
        required_data_columns (list): List of required columns in the dataset.
    Methods:
        __init__(data_path, dataconfig, batch_size=64, max_sequence_length=512, batch_read_file=False, nrows=None, seperator=','):
            Initializes the HeavyChainDataset with the given parameters.
        derive_call_one_hot_representation():
            Derives one-hot encoding representations for V, D, and J alleles.
        derive_call_dictionaries():
            Derives dictionaries for V, D, and J alleles from the dataconfig.
        get_ohe_reverse_mapping():
            Returns the reverse mapping of one-hot encoded alleles.
        _get_single_batch(pointer):
            Retrieves a single batch of data from the dataset and processes it.
        generate_model_params():
            Generates model parameters based on the dataset attributes.
    """
    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=512, use_streaming=False,
                 nrows=None, seperator=','):
        super().__init__(data_path, dataconfig, batch_size, max_sequence_length, use_streaming, nrows, seperator)

        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_call',
                                      'd_call', 'j_call', 'mutation_rate', 'indels', 'productive']

    def derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        j_alleles = sorted(list(self.j_dict))
        # Add Short D Label as Last Label
        d_alleles = d_alleles + ['Short-D']

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.allele_encoder.register_gene("V", v_alleles,sort=False)
        self.allele_encoder.register_gene("D", d_alleles,sort=False)
        self.allele_encoder.register_gene("J", j_alleles,sort=False)


    def derive_call_dictionaries(self):
        self.v_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.v_alleles for j in
                       self.dataconfig.v_alleles[i]}
        self.d_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.d_alleles for j in
                       self.dataconfig.d_alleles[i]}
        self.j_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.j_alleles for j in
                       self.dataconfig.j_alleles[i]}


    def _get_single_batch(self, pointer):
        # Read Batch from Dataset
        batch = self.reader.get_batch(pointer)
        batch = pd.DataFrame(batch)

        # Encoded sequence in batch and collect the padding sizes applied to each sequences
        encoded_sequences, paddings = self.tokenizer.encode_and_pad_center(batch['sequence'])
        # use the padding sizes collected to adjust the start/end positions of the alleles
        for _gene in ['v_sequence', 'd_sequence', 'j_sequence']:
            for _position in ['start', 'end']:
                batch.loc[:, _gene + '_' + _position] += paddings

        x = {"tokenized_sequence": encoded_sequences}

        segments = {'v': [], 'd': [], 'j': []}
        indel_counts = []
        productive = []
        for ax, row in batch.iterrows():
            indels = eval(row['indels'])
            indel_counts.append(len(indels))

        # Convert Comma Seperated Allele Ground Truth Labels into Lists
        v_alleles = batch.v_call.apply(lambda x: set(x.split(',')))
        d_alleles = batch.d_call.apply(lambda x: set(x.split(',')))
        j_alleles = batch.j_call.apply(lambda x: set(x.split(',')))

        y = {
            "v_start": batch.v_sequence_start.values.reshape(-1, 1),
            "v_end": batch.v_sequence_end.values.reshape(-1, 1),
            "d_start": batch.d_sequence_start.values.reshape(-1, 1),
            "d_end": batch.d_sequence_end.values.reshape(-1, 1),
            "j_start": batch.j_sequence_start.values.reshape(-1, 1),
            "j_end": batch.j_sequence_end.values.reshape(-1, 1),
            "v_allele": self.one_hot_encode_allele("V", v_alleles),
            "d_allele": self.one_hot_encode_allele("D", d_alleles),
            "j_allele": self.one_hot_encode_allele("J", j_alleles),
            'mutation_rate': batch.mutation_rate.values.reshape(-1, 1),
            'indel_count': np.array(indel_counts).reshape(-1, 1),
            'productive': np.array([float(eval(i)) for i in batch.productive]).reshape(-1, 1)

        }
        return x, y

    def generate_model_params(self):
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "d_allele_count": self.d_allele_count,
            "j_allele_count": self.j_allele_count,
        }
