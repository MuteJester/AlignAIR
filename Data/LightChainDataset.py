import numpy as np
import pandas as pd

from Data.datasetBase import DatasetBase
from SequenceSimulation.sequence import LightChainType
from SequenceSimulation.utilities import DataConfig


class LightChainDataset(DatasetBase):
    def __init__(self, data_path, lambda_dataconfig: DataConfig,
                 kappa_dataconfig: DataConfig,batch_size=64, max_sequence_length=512, batch_read_file=False,
                 nrows=None, seperator=','):
        super().__init__(data_path, [kappa_dataconfig,lambda_dataconfig], batch_size, max_sequence_length, batch_read_file, nrows, seperator)

        self.dataconfig = [kappa_dataconfig,lambda_dataconfig]
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end',
                                      'j_sequence_start', 'j_sequence_end', 'v_allele',
                                      'j_allele', 'type', 'mutation_rate','indels']

    def derive_call_one_hot_representation(self):
        v_alleles = sorted(list(self.v_kappa_dict)) + sorted(list(self.v_lambda_dict))
        j_alleles = sorted(list(self.j_kappa_dict)) + sorted(list(self.j_lambda_dict))

        self.v_allele_count = len(v_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
        }

    def derive_call_dictionaries(self):
        self.v_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[0].v_alleles for j in
                              self.dataconfig[0].v_alleles[i]}
        self.j_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[0].j_alleles for j in
                              self.dataconfig[0].j_alleles[i]}

        self.v_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[1].v_alleles for j in
                       self.dataconfig[1].v_alleles[i]}
        self.j_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[1].j_alleles for j in
                       self.dataconfig[1].j_alleles[i]}

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
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
        for _gene in ['v_sequence', 'j_sequence']:
            for _position in ['start', 'end']:
                batch.loc[:, _gene + '_' + _position] += paddings

        segments = {'v': [], 'j': []}
        indel_counts = []

        for ax, row in batch.iterrows():
            indels = eval(row['indels'])
            insertions = [i for i in indels if 'I' in indels[i]]
            indel_counts.append(len(indels))
            for _gene in ['v', 'j']:
                empty = np.zeros((1, self.max_seq_length))
                empty[0, row[_gene + '_sequence_start']:row[_gene + '_sequence_end']] = 1
                for I in insertions:
                    empty[0, I] = 0
                segments[_gene].append(empty)

        for _gene in ['v', 'j']:
            segments[_gene] = np.vstack(segments[_gene])

        # Convert Comma Seperated Allele Ground Truth Labels into Lists
        v_alleles = batch.v_allele.apply(lambda x: set(x.split(',')))
        j_alleles = batch.j_allele.apply(lambda x: set(x.split(',')))

        # Convert Type Column into a Binary Representation, 1 = Kappa, 0 = Lambda
        chain_type = [int(i == str(LightChainType.KAPPA)) for i in batch['type']]

        x = {"tokenized_sequence": encoded_sequences}

        y = {
            "v_segment": segments['v'],
            "j_segment": segments['j'],
            "v_allele": self.one_hot_encode_allele("V", v_alleles),
            "j_allele": self.one_hot_encode_allele("J", j_alleles),
            'mutation_rate': batch.mutation_rate.values.reshape(-1, 1),
            'type': np.array(chain_type).reshape(-1, 1),
            'indel_count': np.array(indel_counts).reshape(-1, 1)

        }
        return x, y

    def generate_model_params(self):
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "j_allele_count": self.j_allele_count,
        }



