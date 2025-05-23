import numpy as np
import pandas as pd

from ..Data.datasetBase import DatasetBase
from GenAIRR.sequence import LightChainType
from GenAIRR.dataconfig import DataConfig



class LightChainDataset(DatasetBase):
    def __init__(self, data_path, lambda_dataconfig: DataConfig,
                 kappa_dataconfig: DataConfig, batch_size=64, max_sequence_length=512, use_streaming=False,
                 nrows=None, seperator=','):
        super().__init__(data_path,
               [kappa_dataconfig,lambda_dataconfig],
                         batch_size, max_sequence_length,
                         use_streaming,
                         nrows,
                         seperator,
                         required_data_columns=['sequence', 'v_sequence_start', 'v_sequence_end',
                                      'j_sequence_start', 'j_sequence_end', 'v_call',
                                      'j_call', 'mutation_rate', 'indels', 'productive','type'])

        self.dataconfig = [kappa_dataconfig,lambda_dataconfig]

    def derive_call_one_hot_representation(self):
        v_alleles = sorted(list(self.v_kappa_dict)) + sorted(list(self.v_lambda_dict))
        j_alleles = sorted(list(self.j_kappa_dict)) + sorted(list(self.j_lambda_dict))

        self.v_allele_count = len(v_alleles)
        self.j_allele_count = len(j_alleles)

        self.allele_encoder.register_gene("V", v_alleles,sort=False)
        self.allele_encoder.register_gene("J", j_alleles,sort=False)

    def derive_call_dictionaries(self):
        self.v_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[0].v_alleles for j in
                              self.dataconfig[0].v_alleles[i]}
        self.j_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[0].j_alleles for j in
                              self.dataconfig[0].j_alleles[i]}

        self.v_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[1].v_alleles for j in
                       self.dataconfig[1].v_alleles[i]}
        self.j_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig[1].j_alleles for j in
                       self.dataconfig[1].j_alleles[i]}


    def _get_single_batch(self, pointer):
        # Read Batch from Dataset
        batch = self.reader.get_batch(pointer)
        batch = pd.DataFrame(batch)

        # Encoded sequence in batch and collect the padding sizes applied to each sequences
        encoded_sequences, paddings = self.tokenizer.encode_and_pad_center(batch['sequence'])
        # use the padding sizes collected to adjust the start/end positions of the alleles
        for _gene in ['v_sequence', 'j_sequence']:
            for _position in ['start', 'end']:
                batch.loc[:, _gene + '_' + _position] += paddings

        indel_counts = []
        for ax, row in batch.iterrows():
            indels = eval(row['indels'])
            indel_counts.append(len(indels))

        # Convert Comma Seperated Allele Ground Truth Labels into Lists
        v_alleles = batch.v_call.apply(lambda x: set(x.split(',')))
        j_alleles = batch.j_call.apply(lambda x: set(x.split(',')))

        # Convert Type Column into a Binary Representation, 1 = Kappa, 0 = Lambda
        chain_type = [int(i == str(LightChainType.KAPPA)) for i in batch['type']]

        x = {"tokenized_sequence": encoded_sequences}

        y = {
            "v_start": batch.v_sequence_start.values.reshape(-1, 1),
            "v_end": batch.v_sequence_end.values.reshape(-1, 1),
            "j_start": batch.j_sequence_start.values.reshape(-1, 1),
            "j_end": batch.j_sequence_end.values.reshape(-1, 1),
            "v_allele": self.one_hot_encode_allele("V", v_alleles),
            "j_allele": self.one_hot_encode_allele("J", j_alleles),
            'mutation_rate': batch.mutation_rate.values.reshape(-1, 1),
            'indel_count': np.array(indel_counts).reshape(-1, 1),
            'productive': np.array([float(eval(i)) for i in batch.productive]).reshape(-1, 1),
            'type': np.array(chain_type).reshape(-1, 1),

        }
        return x, y

    def generate_model_params(self):
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "j_allele_count": self.j_allele_count,
        }



