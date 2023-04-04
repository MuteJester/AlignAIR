from airrship.create_repertoire import generate_sequence,load_data,get_genotype
import scipy.stats as st
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

class VDeepJUnbondedDataset():
    def __init__(self, batch_size=64, max_sequence_length=512):
        self.max_sequence_length = max_sequence_length

        self.data_dict = load_data()
        self.locus = get_genotype()
        self.max_seq_length = max_sequence_length
        self.nucleotide_add_distribution = st.beta(1, 3)
        self.nucleotide_remove_distribution = st.beta(1, 3)
        self.add_remove_probability = st.bernoulli(0.5)

        self.tokenizer_dictionary = {
            'A': 1,
            'T': 2,
            'G': 3,
            'C': 4,
            'N': 5,
            'P': 0  # pad token
        }

        self.mutate = True
        self.flat_vdj = True
        self.no_trim_args = False
        self.mutation_rate = 0.08
        self.batch_size = batch_size

        self.derive_call_dictionaries()
        self.derive_counts()
        self.derive_call_one_hot_representation()

    def generate_single(self):
        return generate_sequence(self.locus, self.data_dict, mutate=self.mutate, mutation_rate=self.mutation_rate,
                                 flat_usage='gene')

    def derive_counts(self):

        self.v_family_count = len(set([self.v_dict[i]['family'] for i in self.v_dict]))
        self.v_gene_count = len(set([self.v_dict[i]['gene'] for i in self.v_dict]))
        self.v_allele_count = len(set([self.v_dict[i]['allele'] for i in self.v_dict]))

        self.d_family_count = len(set([self.d_dict[i]['family'] for i in self.d_dict]))
        self.d_gene_count = len(set([self.d_dict[i]['gene'] for i in self.d_dict]))
        self.d_allele_count = len(set([self.d_dict[i]['allele'] for i in self.d_dict]))

        self.j_gene_count = len(set([self.j_dict[i]['gene'] for i in self.j_dict]))
        self.j_allele_count = len(set([self.j_dict[i]['allele'] for i in self.j_dict]))

    def derive_call_one_hot_representation(self):

        v_families = sorted(set([self.v_dict[i]['family'] for i in self.v_dict]))
        v_genes = sorted(set([self.v_dict[i]['gene'] for i in self.v_dict]))
        v_alleles = sorted(set([self.v_dict[i]['allele'] for i in self.v_dict]))

        self.v_family_call_ohe = {f: i for i, f in enumerate(v_families)}
        self.v_gene_call_ohe = {f: i for i, f in enumerate(v_genes)}
        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}

        d_families = sorted(set([self.d_dict[i]['family'] for i in self.d_dict]))
        d_genes = sorted(set([self.d_dict[i]['gene'] for i in self.d_dict]))
        d_alleles = sorted(set([self.d_dict[i]['allele'] for i in self.d_dict]))

        self.d_family_call_ohe = {f: i for i, f in enumerate(d_families)}
        self.d_gene_call_ohe = {f: i for i, f in enumerate(d_genes)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}

        j_genes = sorted(set([self.j_dict[i]['gene'] for i in self.j_dict]))
        j_alleles = sorted(set([self.j_dict[i]['allele'] for i in self.j_dict]))

        self.j_gene_call_ohe = {f: i for i, f in enumerate(j_genes)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

    def derive_call_dictionaries(self):
        self.v_dict, self.d_dict, self.j_dict = dict(), dict(), dict()
        for call in ['V', 'D', 'J']:
            for idx in range(2):
                for N in self.locus[idx][call]:
                    if call == 'V':
                        family, G = N.name.split('-')
                        gene, allele = G.split('*')
                        self.v_dict[N.name] = {'family': family, 'gene': gene, 'allele': allele}
                    elif call == 'D':
                        family, G = N.name.split('-')
                        gene, allele = G.split('*')
                        self.d_dict[N.name] = {'family': family, 'gene': gene, 'allele': allele}
                    elif call == 'J':
                        gene, allele = N.name.split('*')
                        self.j_dict[N.name] = {'gene': gene, 'allele': allele}

    def decompose_call(self, type, call):
        if type == 'V' or type == 'D':
            family, G = call.split('-')
            gene, allele = G.split('*')
            return family, gene, allele
        else:
            return call.split('*')

    def generate_batch(self):
        data = {'sequence': [],
                'v_sequence_start': [],
                'v_sequence_end': [],
                'd_sequence_start': [],
                'd_sequence_end': [],
                'j_sequence_start': [],
                'j_sequence_end': [],
                'v_family': [],
                'v_gene': [],
                'v_allele': [],
                'd_family': [],
                'd_gene': [],
                'd_allele': [],
                'j_gene': [],
                'j_allele': []}

        for _ in range(self.batch_size):
            gen = self.generate_single()
            v_family, v_gene, v_allele = self.decompose_call('V', gen.v_allele.name)
            d_family, d_gene, d_allele = self.decompose_call('D', gen.d_allele.name)
            j_gene, j_allele = self.decompose_call('J', gen.j_allele.name)

            data['sequence'].append(gen.mutated_seq)
            data['v_sequence_start'].append(gen.v_seq_start)
            data['v_sequence_end'].append(gen.v_seq_end)
            data['d_sequence_start'].append(gen.d_seq_start)
            data['d_sequence_end'].append(gen.d_seq_end)
            data['j_sequence_start'].append(gen.j_seq_start)
            data['j_sequence_end'].append(gen.j_seq_end)
            data['v_family'].append(v_family)
            data['v_gene'].append(v_gene)
            data['v_allele'].append(v_allele)
            data['d_family'].append(d_family)
            data['d_gene'].append(d_gene)
            data['d_allele'].append(d_allele)
            data['j_gene'].append(j_gene)
            data['j_allele'].append(j_allele)
        return data

    def _process_and_dpad(self, sequence, train=True):
        """
        Private method, converts sequences into 4 one hot vectors and paddas them from both sides with zeros
        equal to the diffrenece between the max length and the sequence length
        :param nuc:
        :param self.max_seq_length:
        :return:
        """

        start, end = None, None
        trans_seq = [self.tokenizer_dictionary[i] for i in sequence]

        gap = self.max_seq_length - len(trans_seq)
        iseven = gap % 2 == 0
        whole_half_gap = gap // 2

        if iseven:
            trans_seq = [0] * whole_half_gap + trans_seq + ([0] * whole_half_gap)
            if train:
                start, end = whole_half_gap, self.max_seq_length - whole_half_gap - 1

        else:
            trans_seq = [0] * (whole_half_gap + 1) + trans_seq + ([0] * whole_half_gap)
            if train:
                start, end = whole_half_gap + 1, self.max_seq_length - whole_half_gap - 1

        return trans_seq, start, end if iseven else (end + 1)

    def process_sequences(self, data: pd.DataFrame, corrupt_beginning=False, verbose=False):
        padded_sequences = []
        v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []

        if verbose:
            iterator = tqdm(data.itertuples(), total=len(data))
        else:
            iterator = data.itertuples()

        for row in iterator:
            if corrupt_beginning is True:
                seq, was_removed, amount_changed = self._corrupt_sequence_beginning(row.sequence)
            else:
                seq = row.sequence

            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)

            if corrupt_beginning:
                if was_removed:
                    # v is shorter
                    _adjust = start - amount_changed
                else:
                    # v is longer
                    _adjust = start + amount_changed
                    start += amount_changed
            else:
                _adjust = start

            v_start.append(start)
            j_end.append(end)
            v_end.append(row.v_sequence_end + _adjust)
            d_start.append(row.d_sequence_start + _adjust)
            d_end.append(row.d_sequence_end + _adjust)
            j_start.append(row.j_sequence_start + _adjust)

        v_start = np.array(v_start)
        v_end = np.array(v_end)
        d_start = np.array(d_start)
        d_end = np.array(d_end)
        j_start = np.array(j_start)
        j_end = np.array(j_end)

        padded_sequences = np.vstack(padded_sequences)

        return v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences

    def get_ohe(self, type, level, values):
        if type == 'V':
            if level == 'family':
                result = []
                for value in values:
                    ohe = np.zeros(self.v_family_count)
                    ohe[self.v_family_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
            elif level == 'gene':
                result = []
                for value in values:
                    ohe = np.zeros(self.v_gene_count)
                    ohe[self.v_gene_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
            else:
                result = []
                for value in values:
                    ohe = np.zeros(self.v_allele_count)
                    ohe[self.v_allele_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
        elif type == 'D':
            if level == 'family':
                result = []
                for value in values:
                    ohe = np.zeros(self.d_family_count)
                    ohe[self.d_family_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
            elif level == 'gene':
                result = []
                for value in values:
                    ohe = np.zeros(self.d_gene_count)
                    ohe[self.d_gene_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
            else:
                result = []
                for value in values:
                    ohe = np.zeros(self.d_allele_count)
                    ohe[self.d_allele_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
        else:
            if level == 'gene':
                result = []
                for value in values:
                    ohe = np.zeros(self.j_gene_count)
                    ohe[self.j_gene_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
            else:
                result = []
                for value in values:
                    ohe = np.zeros(self.j_allele_count)
                    ohe[self.j_allele_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)

    def _get_single_batch(self):
        data = pd.DataFrame(self.generate_batch())
        v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences = self.process_sequences(
            data, corrupt_beginning=True)
        x = {'tokenized_sequence': padded_sequences, 'tokenized_sequence_for_masking': padded_sequences}
        y = {'v_start': v_start, 'v_end': v_end, 'd_start': d_start, 'd_end': d_end, 'j_start': j_start,
             'j_end': j_end,
             'v_family': self.get_ohe('V', 'family', data.v_family),
             'v_gene': self.get_ohe('V', 'gene', data.v_gene),
             'v_allele': self.get_ohe('V', 'allele', data.v_allele),
             'd_family': self.get_ohe('D', 'family', data.d_family),
             'd_gene': self.get_ohe('D', 'gene', data.d_gene),
             'd_allele': self.get_ohe('D', 'allele', data.d_allele),
             'j_gene': self.get_ohe('J', 'gene', data.j_gene),
             'j_allele': self.get_ohe('J', 'allele', data.j_allele),

             }
        return x, y

    def _get_tf_dataset_params(self):

        x, y = self._get_single_batch()

        output_types = ({k: tf.float32 for k in x}, {k: tf.float32 for k in y})

        output_shapes = ({k: (self.batch_size,) + x[k].shape[1:] for k in x},
                         {k: (self.batch_size,) + y[k].shape[1:] for k in y})

        return output_types, output_shapes

    def _generate_random_nucleotide_sequence(self, length):
        sequence = ''.join(np.random.choice(['A', 'T', 'C', 'G'], size=length))
        return sequence

    def _fix_sequence_validity_after_corruption(self, sequence):
        if len(sequence) > self.max_seq_length:
            # In case we added too many nucleotide to the beginning while corrupting the sequence, remove the slack
            slack = len(sequence) - self.max_seq_length
            return sequence[slack:]
        else:
            return sequence

    def _corrupt_sequence_beginning(self, sequence):
        to_remove = self._sample_add_remove_distribution()
        if to_remove:
            amount_to_remove = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_remove:]
            return modified_sequence, to_remove, amount_to_remove

        else:
            amount_to_add = self._sample_nucleotide_add_distribution(1)
            modified_sequence = self._generate_random_nucleotide_sequence(amount_to_add)
            modified_sequence = modified_sequence + sequence
            modified_sequence = self._fix_sequence_validity_after_corruption(modified_sequence)
            return modified_sequence, to_remove, amount_to_add

    def _train_generator(self):
        while True:
            batch_x, batch_y = self._get_single_batch()
            yield batch_x, batch_y

    def get_train_dataset(self):

        output_types, output_shapes = self._get_tf_dataset_params()

        dataset = tf.data.Dataset.from_generator(
            lambda: self._train_generator(),
            output_types=output_types,
            output_shapes=output_shapes
        )

        return dataset

    def generate_model_params(self):
        return {
            'max_seq_length': self.max_sequence_length,
            'v_family_count': self.v_family_count,
            'v_gene_count': self.v_gene_count,
            'v_allele_count': self.v_allele_count,
            'd_family_count': self.d_family_count,
            'd_gene_count': self.d_gene_count,
            'd_allele_count': self.d_allele_count,
            'j_gene_count': self.j_gene_count,
            'j_allele_count': self.j_allele_count
        }

    def _sample_nucleotide_add_distribution(self, size):
        sample = (35 * self.nucleotide_add_distribution.rvs(size=size)).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _sample_nucleotide_remove_distribution(self, size):
        sample = (50 * self.nucleotide_remove_distribution.rvs(size=size)).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _sample_add_remove_distribution(self):
        return bool(self.add_remove_probability.rvs(1))

    def __len__(self):
        return len(self.data)