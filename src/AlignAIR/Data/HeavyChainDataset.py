import numpy as np
import pandas as pd

from ..Data.datasetBase import DatasetBase
from GenAIRR.utilities import DataConfig


class HeavyChainDataset(DatasetBase):
    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=512, batch_read_file=False,
                 nrows=None, seperator=','):
        super().__init__(data_path, dataconfig, batch_size, max_sequence_length, batch_read_file, nrows, seperator)

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

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe}
        }

    def derive_call_dictionaries(self):
        self.v_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.v_alleles for j in
                       self.dataconfig.v_alleles[i]}
        self.d_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.d_alleles for j in
                       self.dataconfig.d_alleles[i]}
        self.j_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.j_alleles for j in
                       self.dataconfig.j_alleles[i]}

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def _get_single_batch(self, pointer):
        # Read Batch from Dataset
        batch = self.generate_batch(pointer)
        batch = pd.DataFrame(batch)

        # Encoded sequence in batch and collect the padding sizes applied to each sequences
        encoded_sequences, paddings = self.encode_and_pad_sequences(batch['sequence'])
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

