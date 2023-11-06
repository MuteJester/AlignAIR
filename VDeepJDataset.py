import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
from SequenceCorruptor import SequenceCorruptor, SequenceCorruptorV2, SequenceCorruptorV2
from VDeepJUnbondedDataset import global_genotype
import tensorflow as tf
import scipy.stats as st
from collections import defaultdict
import csv


def cast_if_int(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return element
    try:
        f = int(element)
        return f
    except ValueError:
        return element


def count_tsv_size(path):
    with open(path) as fp:
        for (count, _) in enumerate(fp, 1):
            pass
        return count - 1


def table_generator(path, batch_size, usecols=None, loop=True, seperator='\t'):
    while True:
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=seperator)
            headers = next(reader)  # Skip header row
            if len(set(usecols) & set(headers)) != len(usecols):
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

            if len(batch) > 0:
                yield batch
                if not loop:
                    break


class VDeepJDataset:
    def __init__(self, data_path, max_sequence_length=512, batch_size=64, nrows=None, corrupt_beginning=False,
                 corrupt_proba=1, nucleotide_add_coef=35, nucleotide_remove_coef=50, batch_read_file=False):
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.batch_read_file = batch_read_file
        self.required_data_columns = ['sequence', 'v_call', 'd_call', 'j_call', 'v_sequence_end',
                                      'd_sequence_start',
                                      'j_sequence_start',
                                      'j_sequence_end', 'd_sequence_end', 'v_sequence_start']

        if not self.batch_read_file:
            self.data = pd.read_table(data_path,
                                      usecols=self.required_data_columns, nrows=nrows)
            self.data_length = len(self.data)
        else:
            self.get_data_batch_generator(data_path)
            self.data_length = count_tsv_size(data_path)

        self.nucleotide_add_distribution = st.beta(1, 3)
        self.nucleotide_remove_distribution = st.beta(1, 3)
        self.add_remove_probability = st.bernoulli(0.5)
        self.corrupt_beginning = corrupt_beginning
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.max_seq_length = max_sequence_length
        self.tokenizer_dictionary = {
            'A': 1,
            'T': 2,
            'G': 3,
            'C': 4,
            'N': 5,
            'P': 0  # pad token
        }
        self.corrupt_beginning = corrupt_beginning
        self.locus = global_genotype()
        self.derive_call_dictionaries()
        self.derive_counts()
        self.derive_call_one_hot_representation()
        self.derive_reverse_ohe_mapping()
        self._calculate_sub_classes_map()
        self._derive_sub_classes_dict()

        # self.derive_call_sections()

    def get_data_batch_generator(self, path_to_data):
        self.data = table_generator(path_to_data, batch_size=self.batch_size, usecols=self.required_data_columns)

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

        self.label_num_sub_classes_dict = {
            "V": {
                "family": self.v_family_call_ohe,
                "gene": self.v_gene_call_ohe,
                "allele": self.v_allele_call_ohe,
            },
            "D": {
                "family": self.d_family_call_ohe,
                "gene": self.d_gene_call_ohe,
                "allele": self.d_allele_call_ohe,
            },
            "J": {
                "gene": self.j_gene_call_ohe,
                "allele": self.j_allele_call_ohe,
            },
        }

    def derive_call_sections(self):
        self.v_family = [self.v_dict[x]["family"] for x in tqdm(self.data.v_call)]
        self.d_family = [self.d_dict[x]["family"] for x in tqdm(self.data.d_call)]

        self.v_gene = [self.v_dict[x]["gene"] for x in tqdm(self.data.v_call)]
        self.d_gene = [self.d_dict[x]["gene"] for x in tqdm(self.data.d_call)]
        self.j_gene = [self.j_dict[x]["gene"] for x in tqdm(self.data.j_call)]

        self.v_allele = [self.v_dict[x]["allele"] for x in tqdm(self.data.v_call)]
        self.d_allele = [self.d_dict[x]["allele"] for x in tqdm(self.data.d_call)]
        self.j_allele = [self.j_dict[x]["allele"] for x in tqdm(self.data.j_call)]

    @staticmethod
    def reverse_dictionary(dictionary):
        return {i: j for j, i in dictionary.items()}

    def derive_call_dictionaries(self):
        self.v_dict, self.d_dict, self.j_dict = dict(), dict(), dict()
        for call in ['V', 'D', 'J']:
            for idx in range(2):
                for N in self.locus[idx][call]:
                    if call == 'V':
                        family, G = N.name.split('-', 1)
                        gene, allele = G.split('*')
                        self.v_dict[N.name] = {'family': family, 'gene': gene, 'allele': allele}
                    elif call == 'D':
                        family, G = N.name.split('-', 1)
                        gene, allele = G.split('*')
                        self.d_dict[N.name] = {'family': family, 'gene': gene, 'allele': allele}
                    elif call == 'J':
                        gene, allele = N.name.split('*', 1)
                        self.j_dict[N.name] = {'gene': gene, 'allele': allele}

    def derive_reverse_ohe_mapping(self):
        self.reverse_ohe_mapping = {
            'v_family': self.reverse_dictionary(self.v_family_call_ohe),
            'v_gene': self.reverse_dictionary(self.v_gene_call_ohe),
            'v_allele': self.reverse_dictionary(self.v_allele_call_ohe),
            'd_family': self.reverse_dictionary(self.d_family_call_ohe),
            'd_gene': self.reverse_dictionary(self.d_gene_call_ohe),
            'd_allele': self.reverse_dictionary(self.d_allele_call_ohe),
            'j_gene': self.reverse_dictionary(self.j_gene_call_ohe),
            'j_allele': self.reverse_dictionary(self.j_allele_call_ohe),

        }

    def decompose_call(self, type, call):
        if type == 'V' or type == 'D':
            family, G = call.split('-')
            gene, allele = G.split('*')
            return family, gene, allele
        else:
            return call.split('*')

    def generate_model_params(self):
        return {
            'max_seq_length': self.max_seq_length,
            'v_family_count': self.v_family_count,
            'v_gene_count': self.v_gene_count,
            'v_allele_count': self.v_allele_count,
            'd_family_count': self.d_family_count,
            'd_gene_count': self.d_gene_count,
            'd_allele_count': self.d_allele_count,
            'j_gene_count': self.j_gene_count,
            'j_allele_count': self.j_allele_count
        }

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
            to_corrupt = bool(np.random.binomial(1, self.corrupt_proba))
            if corrupt_beginning and to_corrupt:
                seq, was_removed, amount_changed = self._corrupt_sequence_beginning(row.sequence)
            else:
                seq = row.sequence

            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)

            if corrupt_beginning and to_corrupt:
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

    def tokenize_sequences(self, sequences, verbose=False):
        padded_sequences = []

        if verbose:
            iterator = tqdm(sequences, total=len(sequences))
        else:
            iterator = sequences

        for seq in iterator:
            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)
        padded_sequences = np.vstack(padded_sequences)

        return padded_sequences

    def generate_batch(self, pointer):
        if not self.batch_read_file:
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
            for idx, row in self.data.iloc[(pointer - self.batch_size):pointer, :].iterrows():
                v_family, v_gene, v_allele = self.decompose_call('V', row['v_call'])
                d_family, d_gene, d_allele = self.decompose_call('D', row['d_call'])
                j_gene, j_allele = self.decompose_call('J', row['j_call'])

                data['sequence'].append(row['sequence'])
                data['v_sequence_start'].append(row['v_sequence_start'])
                data['v_sequence_end'].append(row['v_sequence_end'])
                data['d_sequence_start'].append(row['d_sequence_start'])
                data['d_sequence_end'].append(row['d_sequence_end'])
                data['j_sequence_start'].append(row['j_sequence_start'])
                data['j_sequence_end'].append(row['j_sequence_end'])
                data['v_family'].append(v_family)
                data['v_gene'].append(v_gene)
                data['v_allele'].append(v_allele)
                data['d_family'].append(d_family)
                data['d_gene'].append(d_gene)
                data['d_allele'].append(d_allele)
                data['j_gene'].append(j_gene)
                data['j_allele'].append(j_allele)

            return data
        else:
            read_batch = next(self.data)
            for missing_column in ['v_family', 'v_gene', 'v_allele',
                                   'd_family', 'd_gene', 'd_allele',
                                   'j_gene', 'j_allele']:
                read_batch[missing_column] = []

            for idx in range(self.batch_size):
                v_family, v_gene, v_allele = self.decompose_call('V', read_batch['v_call'][idx])
                d_family, d_gene, d_allele = self.decompose_call('D', read_batch['d_call'][idx])
                j_gene, j_allele = self.decompose_call('J', read_batch['j_call'][idx])

                read_batch['v_family'].append(v_family)
                read_batch['v_gene'].append(v_gene)
                read_batch['v_allele'].append(v_allele)
                read_batch['d_family'].append(d_family)
                read_batch['d_gene'].append(d_gene)
                read_batch['d_allele'].append(d_allele)
                read_batch['j_gene'].append(j_gene)
                read_batch['j_allele'].append(j_allele)

            return read_batch

    def _get_single_batch(self, pointer):

        batch = self.generate_batch(pointer)
        data = pd.DataFrame(batch)

        v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences = self.process_sequences(
            data, corrupt_beginning=self.corrupt_beginning)

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

    def _train_generator(self):
        pointer = 0
        while True:
            pointer += self.batch_size
            if pointer >= self.data_length:
                pointer = self.batch_size

            batch_x, batch_y = self._get_single_batch(pointer)

            yield batch_x, batch_y

    def get_train_dataset(self):

        output_types, output_shapes = self._get_tf_dataset_params()

        dataset = tf.data.Dataset.from_generator(
            lambda: self._train_generator(),
            output_types=output_types,
            output_shapes=output_shapes
        )

        return dataset

    def _get_tf_dataset_params(self):

        x, y = self._get_single_batch(self.batch_size)

        output_types = ({k: tf.float32 for k in x}, {k: tf.float32 for k in y})

        output_shapes = ({k: (self.batch_size,) + x[k].shape[1:] for k in x},
                         {k: (self.batch_size,) + y[k].shape[1:] for k in y})

        return output_types, output_shapes

    def generate_model_params(self):
        return {
            "max_seq_length": self.max_sequence_length,
            "v_family_count": self.v_family_count,
            "v_gene_count": self.v_gene_count,
            "v_allele_count": self.v_allele_count,
            "d_family_count": self.d_family_count,
            "d_gene_count": self.d_gene_count,
            "d_allele_count": self.d_allele_count,
            "j_gene_count": self.j_gene_count,
            "j_allele_count": self.j_allele_count,
            "ohe_sub_classes_dict": self.ohe_sub_classes_dict,
        }

    def create_vector(self, idxs, length):
        vector = np.zeros(length)
        vector[idxs] = 1
        return vector

    def _calculate_sub_classes_map(self):
        self.call_sub_classes_map = {"V": defaultdict(lambda: defaultdict(dict)),
                                     "D": defaultdict(lambda: defaultdict(dict)),
                                     "J": defaultdict(dict)}

        def nested_get_sub_classes(d, v_d_or_j):
            for call in d.values():
                if v_d_or_j in ["V", "D"]:
                    fam, gene, allele = call["family"], call["gene"], call["allele"]
                    self.call_sub_classes_map[v_d_or_j][fam][gene][allele] = allele
                elif v_d_or_j == "J":
                    gene, allele = call["gene"], call["allele"]
                    self.call_sub_classes_map[v_d_or_j][gene][allele] = allele

        nested_get_sub_classes(self.v_dict, "V")
        nested_get_sub_classes(self.d_dict, "D")
        nested_get_sub_classes(self.j_dict, "J")

    def _derive_sub_classes_dict(self):
        self.ohe_sub_classes_dict = {
            "V": {"family": {}, "gene": {}},
            "D": {"family": {}, "gene": {}},
            "J": {"gene": {}},
        }
        counts_dict = {
            "V_family_count": self.v_family_count,
            "V_gene_count": self.v_gene_count,
            "V_allele_count": self.v_allele_count,
            "D_family_count": self.d_family_count,
            "D_gene_count": self.d_gene_count,
            "D_allele_count": self.d_allele_count,
            "J_gene_count": self.j_gene_count,
            "J_allele_count": self.j_allele_count,
        }
        for v_d_or_j in ["V", "D", "J"]:
            # Calculate the masked gene vectors for families
            if v_d_or_j in ["V", "D"]:
                for fam in self.call_sub_classes_map[v_d_or_j].keys():
                    label_num = self.label_num_sub_classes_dict[v_d_or_j]["family"][fam]
                    wanted_keys = list(
                        self.call_sub_classes_map[v_d_or_j][fam].keys()
                    )
                    possible_idxs = []
                    for wanted_key in wanted_keys:
                        possible_idxs.append(
                            self.label_num_sub_classes_dict[v_d_or_j]["gene"][
                                wanted_key
                            ]
                        )
                    self.ohe_sub_classes_dict[v_d_or_j]["family"][
                        label_num
                    ] = self.create_vector(
                        possible_idxs, counts_dict[v_d_or_j + "_gene_count"]
                    )
                # Calculate the masked allels vectors for genes
                for fam, fam_dict in self.call_sub_classes_map[v_d_or_j].items():
                    fam_label_num = self.label_num_sub_classes_dict[v_d_or_j]["family"][
                        fam
                    ]
                    self.ohe_sub_classes_dict[v_d_or_j]["gene"][fam_label_num] = {}
                    for gene in fam_dict.keys():
                        if (
                                gene
                                not in self.ohe_sub_classes_dict[v_d_or_j]["gene"][
                            fam_label_num
                        ].keys()
                        ):
                            label_num = self.label_num_sub_classes_dict[v_d_or_j][
                                "gene"
                            ][gene]
                            wanted_keys = list(
                                self.call_sub_classes_map[v_d_or_j][fam][
                                    gene
                                ].keys()
                            )
                            possible_idxs = []
                            for wanted_key in wanted_keys:
                                possible_idxs.append(
                                    self.label_num_sub_classes_dict[v_d_or_j]["allele"][
                                        wanted_key
                                    ]
                                )
                            self.ohe_sub_classes_dict[v_d_or_j]["gene"][fam_label_num][
                                label_num
                            ] = self.create_vector(
                                possible_idxs, counts_dict[v_d_or_j + "_allele_count"]
                            )
            else:  # J
                # Calculate the masked allels vectors for genes
                for gene in self.call_sub_classes_map[v_d_or_j].keys():
                    if gene not in self.ohe_sub_classes_dict[v_d_or_j]["gene"].keys():
                        label_num = self.label_num_sub_classes_dict[v_d_or_j]["gene"][
                            gene
                        ]
                        wanted_keys = list(
                            self.call_sub_classes_map[v_d_or_j][gene].keys()
                        )
                        possible_idxs = []
                        for wanted_key in wanted_keys:
                            possible_idxs.append(
                                self.label_num_sub_classes_dict[v_d_or_j]["allele"][
                                    wanted_key
                                ]
                            )
                        self.ohe_sub_classes_dict[v_d_or_j]["gene"][
                            label_num
                        ] = self.create_vector(
                            possible_idxs, counts_dict[v_d_or_j + "_allele_count"]
                        )

        ### Just for assertion, can be removed later ###
        v_family_keys = list(self.ohe_sub_classes_dict["V"]["family"].keys())
        v_family_keys.sort()
        assert v_family_keys == list(range(self.v_family_count))

        v_gene_keys = list(self.ohe_sub_classes_dict["V"]["gene"].keys())
        v_gene_keys.sort()
        assert v_gene_keys == list(range(self.v_family_count))

        d_family_keys = list(self.ohe_sub_classes_dict["D"]["family"].keys())
        d_family_keys.sort()
        assert d_family_keys == list(range(self.d_family_count))

        d_gene_keys = list(self.ohe_sub_classes_dict["D"]["gene"].keys())
        d_gene_keys.sort()
        assert d_gene_keys == list(range(self.d_family_count))

        j_gene_keys = list(self.ohe_sub_classes_dict["J"]["gene"].keys())
        j_gene_keys.sort()
        assert j_gene_keys == list(range(self.j_gene_count))
        ###

        # Stack all vectors (by index order) for useful indexing
        # V
        self.ohe_sub_classes_dict["V"]["family"] = np.stack(
            [
                self.ohe_sub_classes_dict["V"]["family"][i]
                for i in list(range(self.v_family_count))
            ]
        )
        self.ohe_sub_classes_dict["V"]["family"] = tf.convert_to_tensor(
            self.ohe_sub_classes_dict["V"]["family"], dtype=tf.float32
        )

        tmp_tensor = np.zeros(
            (self.v_family_count, self.v_gene_count, self.v_allele_count)
        )
        for fam_label_num in self.ohe_sub_classes_dict["V"]["gene"].keys():
            for gene_label_num in self.ohe_sub_classes_dict["V"]["gene"][
                fam_label_num
            ].keys():
                tmp_tensor[fam_label_num, gene_label_num] = self.ohe_sub_classes_dict[
                    "V"
                ]["gene"][fam_label_num][gene_label_num]
        tmp_tensor = tf.convert_to_tensor(tmp_tensor, dtype=tf.float32)
        self.ohe_sub_classes_dict["V"]["gene"] = tmp_tensor

        # D
        self.ohe_sub_classes_dict["D"]["family"] = np.stack(
            [
                self.ohe_sub_classes_dict["D"]["family"][i]
                for i in list(range(self.d_family_count))
            ]
        )
        self.ohe_sub_classes_dict["D"]["family"] = tf.convert_to_tensor(
            self.ohe_sub_classes_dict["D"]["family"], dtype=tf.float32
        )

        tmp_tensor = np.zeros(
            (self.d_family_count, self.d_gene_count, self.d_allele_count)
        )
        for fam_label_num in self.ohe_sub_classes_dict["D"]["gene"].keys():
            for gene_label_num in self.ohe_sub_classes_dict["D"]["gene"][
                fam_label_num
            ].keys():
                tmp_tensor[fam_label_num, gene_label_num] = self.ohe_sub_classes_dict[
                    "D"
                ]["gene"][fam_label_num][gene_label_num]
        tmp_tensor = tf.convert_to_tensor(tmp_tensor, dtype=tf.float32)
        self.ohe_sub_classes_dict["D"]["gene"] = tmp_tensor

        # J
        self.ohe_sub_classes_dict["J"]["gene"] = np.stack(
            [
                self.ohe_sub_classes_dict["J"]["gene"][i]
                for i in list(range(self.j_gene_count))
            ]
        )
        self.ohe_sub_classes_dict["J"]["gene"] = tf.convert_to_tensor(
            self.ohe_sub_classes_dict["J"]["gene"], dtype=tf.float32
        )

    def __len__(self):
        return len(self.data)


class VDeepJDatasetSingleBeam():
    def __init__(self, data_path, batch_size=64, max_sequence_length=512, batch_read_file=False, nrows=None,
                 mutation_rate=0.08, shm_flat=False,
                 randomize_rate=False,
                 corrupt_beginning=True, corrupt_proba=1, nucleotide_add_coef=35, nucleotide_remove_coef=50,
                 mutation_oracle_mode=False,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, allele_map_path='/home/bcrlab/thomas/AlignAIRR/',
                 seperator=','):
        self.max_sequence_length = max_sequence_length

        self.locus = global_genotype()
        self.max_seq_length = max_sequence_length
        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.add_remove_probability = st.bernoulli(0.5)
        self.corrupt_beginning = corrupt_beginning
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.mutation_oracle_mode = mutation_oracle_mode
        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP.pkl', 'rb') as h:
            self.VA = pickle.load(h)
            self.MAX_VA = max(self.VA[list(self.VA)[0]])

        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP_AT_END.pkl', 'rb') as h:
            self.VEND_SIM_MAP = pickle.load(h)
            self.MAX_VEND = max(self.VA[list(self.VA)[0]])

        self.sequence_corruptor = SequenceCorruptor(
            nucleotide_add_coef=nucleotide_add_coef, nucleotide_remove_coef=nucleotide_remove_coef, max_length=512,
            random_sequence_add_proba=random_sequence_add_proba, single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba,
            corrupt_proba=self.corrupt_proba,
            nucleotide_add_distribution=self.nucleotide_add_distribution,
            nucleotide_remove_distribution=self.nucleotide_remove_distribution

        )

        self.seperator = seperator
        self.batch_read_file = batch_read_file
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_allele',
                                      'd_allele', 'j_allele']

        self.mutate = True
        self.flat_vdj = True
        self.randomize_rate = randomize_rate
        self.no_trim_args = False
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.shm_flat = shm_flat
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()

        if not self.batch_read_file:
            self.data = pd.read_table(data_path,
                                      usecols=self.required_data_columns, nrows=nrows, sep=self.seperator)
            self.data_length = len(self.data)
        else:
            self.get_data_batch_generator(data_path)
            self.data_length = count_tsv_size(data_path)

    def get_data_batch_generator(self, path_to_data):
        self.data = table_generator(path_to_data, batch_size=self.batch_size, usecols=self.required_data_columns,
                                    seperator=self.seperator)

    def derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        j_alleles = sorted(list(self.j_dict))

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
        }

    def derive_call_dictionaries(self):
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['V']}
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['D']}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['J']}

    def generate_batch(self, pointer):
        if not self.batch_read_file:
            data = {
                "sequence": [],
                "v_sequence_start": [],
                "v_sequence_end": [],
                "d_sequence_start": [],
                "d_sequence_end": [],
                "j_sequence_start": [],
                "j_sequence_end": [],
                "v_allele": [],
                "d_allele": [],
                "j_allele": [],
            }
            for idx, row in self.data.iloc[(pointer - self.batch_size):pointer, :].iterrows():
                data['sequence'].append(row['sequence'])
                data['v_sequence_start'].append(row['v_sequence_start'])
                data['v_sequence_end'].append(row['v_sequence_end'])
                data['d_sequence_start'].append(row['d_sequence_start'])
                data['d_sequence_end'].append(row['d_sequence_end'])
                data['j_sequence_start'].append(row['j_sequence_start'])
                data['j_sequence_end'].append(row['j_sequence_end'])
                data['v_allele'].append(row['v_allele'])
                data['d_allele'].append(row['d_allele'])
                data['j_allele'].append(row['j_allele'])

            return data
        else:
            read_batch = next(self.data)
            return read_batch

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
                start, end = (
                    whole_half_gap + 1,
                    self.max_seq_length - whole_half_gap - 1,
                )

        return trans_seq, start, end if iseven else (end + 1)

    def process_sequences(
            self, data: pd.DataFrame, corrupt_beginning=False, verbose=False
    ):
        return self.sequence_corruptor.process_sequences(data=data, corrupt_beginning=corrupt_beginning,
                                                         verbose=verbose)

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def get_ohe(self, type, values):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []
        for value in values:
            ohe = np.zeros(allele_count)
            ohe[allele_call_ohe[value]] = 1
            result.append(ohe)
        return np.vstack(result)

    def get_expanded_ohe(self, type, values, removed, ends_at=None):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []

        for value, remove, end in zip(values, removed, ends_at):
            ohe = np.zeros(allele_count)

            # get all alleles that are equally likely due missing nucleotides in the beginning
            equal_alleles = self.VA[value][min(remove, self.MAX_VA)]
            # get all allele that are equally likely due to missing nucleotides in the end
            equal_alleles += self.VEND_SIM_MAP[value][min(end, self.MAX_VEND)]

            for i in equal_alleles:
                ohe[allele_call_ohe[i]] = 1
            result.append(ohe)
        return np.vstack(result)

    def _get_single_batch(self, pointer):
        batch = self.generate_batch(pointer)
        data = pd.DataFrame(batch)
        original_ends = data.v_sequence_end
        (
            v_start,
            v_end,
            d_start,
            d_end,
            j_start,
            j_end,
            padded_sequences,
        ) = self.process_sequences(data, corrupt_beginning=self.corrupt_beginning)

        removed = original_ends - (v_end - v_start)
        x = {
            "tokenized_sequence": padded_sequences,
            "tokenized_sequence_for_masking": padded_sequences,
        }
        y = {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_allele": self.get_expanded_ohe("V", data.v_allele, removed, original_ends),
            "d_allele": self.get_ohe("D", data.d_allele),
            "j_allele": self.get_ohe("J", data.j_allele),
        }
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
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "d_allele_count": self.d_allele_count,
            "j_allele_count": self.j_allele_count,
        }

    def tokenize_sequences(self, sequences, verbose=False):
        padded_sequences = []

        if verbose:
            iterator = tqdm(sequences, total=len(sequences))
        else:
            iterator = sequences

        for seq in iterator:
            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)
        padded_sequences = np.vstack(padded_sequences)

        return padded_sequences

    def __len__(self):
        return len(self.data)


class VDeepJDatasetSingleBeamSegmentation():
    def __init__(self, data_path, batch_size=64, max_sequence_length=512, batch_read_file=False, nrows=None,
                 mutation_rate=0.08, shm_flat=False,
                 randomize_rate=False,
                 corrupt_beginning=True, corrupt_proba=1, nucleotide_add_coef=35, nucleotide_remove_coef=50,
                 mutation_oracle_mode=False,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, allele_map_path='/home/bcrlab/thomas/AlignAIRR/',
                 seperator=','):
        self.max_sequence_length = max_sequence_length

        self.locus = global_genotype()
        self.max_seq_length = max_sequence_length
        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.add_remove_probability = st.bernoulli(0.5)
        self.corrupt_beginning = corrupt_beginning
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.mutation_oracle_mode = mutation_oracle_mode
        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP.pkl', 'rb') as h:
            self.VA = pickle.load(h)
            self.MAX_VA = max(self.VA[list(self.VA)[0]])

        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP_AT_END.pkl', 'rb') as h:
            self.VEND_SIM_MAP = pickle.load(h)
            self.MAX_VEND = max(self.VA[list(self.VA)[0]])

        self.sequence_corruptor = SequenceCorruptor(
            nucleotide_add_coef=nucleotide_add_coef, nucleotide_remove_coef=nucleotide_remove_coef, max_length=512,
            random_sequence_add_proba=random_sequence_add_proba, single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba,
            corrupt_proba=self.corrupt_proba,
            nucleotide_add_distribution=self.nucleotide_add_distribution,
            nucleotide_remove_distribution=self.nucleotide_remove_distribution

        )

        self.seperator = seperator
        self.batch_read_file = batch_read_file
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_allele',
                                      'd_allele', 'j_allele']

        self.mutate = True
        self.flat_vdj = True
        self.randomize_rate = randomize_rate
        self.no_trim_args = False
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.shm_flat = shm_flat
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()

        if not self.batch_read_file:
            self.data = pd.read_table(data_path,
                                      usecols=self.required_data_columns, nrows=nrows, sep=self.seperator)
            self.data_length = len(self.data)
        else:
            self.get_data_batch_generator(data_path)
            self.data_length = count_tsv_size(data_path)

    def get_data_batch_generator(self, path_to_data):
        self.data = table_generator(path_to_data, batch_size=self.batch_size, usecols=self.required_data_columns,
                                    seperator=self.seperator)

    def derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        j_alleles = sorted(list(self.j_dict))

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
        }

    def derive_call_dictionaries(self):
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['V']}
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['D']}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['J']}

    def generate_batch(self, pointer):
        if not self.batch_read_file:
            data = {
                "sequence": [],
                "v_sequence_start": [],
                "v_sequence_end": [],
                "d_sequence_start": [],
                "d_sequence_end": [],
                "j_sequence_start": [],
                "j_sequence_end": [],
                "v_allele": [],
                "d_allele": [],
                "j_allele": [],
            }
            for idx, row in self.data.iloc[(pointer - self.batch_size):pointer, :].iterrows():
                data['sequence'].append(row['sequence'])
                data['v_sequence_start'].append(row['v_sequence_start'])
                data['v_sequence_end'].append(row['v_sequence_end'])
                data['d_sequence_start'].append(row['d_sequence_start'])
                data['d_sequence_end'].append(row['d_sequence_end'])
                data['j_sequence_start'].append(row['j_sequence_start'])
                data['j_sequence_end'].append(row['j_sequence_end'])
                data['v_allele'].append(row['v_allele'])
                data['d_allele'].append(row['d_allele'])
                data['j_allele'].append(row['j_allele'])

            return data
        else:
            read_batch = next(self.data)
            return read_batch

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
                start, end = (
                    whole_half_gap + 1,
                    self.max_seq_length - whole_half_gap - 1,
                )

        return trans_seq, start, end if iseven else (end + 1)

    def process_sequences(
            self, data: pd.DataFrame, corrupt_beginning=False, verbose=False
    ):
        return self.sequence_corruptor.process_sequences(data=data, corrupt_beginning=corrupt_beginning,
                                                         verbose=verbose)

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def get_ohe(self, type, values):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []
        for value in values:
            ohe = np.zeros(allele_count)
            ohe[allele_call_ohe[value]] = 1
            result.append(ohe)
        return np.vstack(result)

    def get_expanded_ohe(self, type, values, removed, ends_at=None):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []

        for value, remove, end in zip(values, removed, ends_at):
            ohe = np.zeros(allele_count)

            # get all alleles that are equally likely due missing nucleotides in the beginning
            equal_alleles = self.VA[value][min(remove, self.MAX_VA)]
            # get all allele that are equally likely due to missing nucleotides in the end
            equal_alleles += self.VEND_SIM_MAP[value][min(end, self.MAX_VEND)]

            for i in equal_alleles:
                ohe[allele_call_ohe[i]] = 1
            result.append(ohe)
        return np.vstack(result)

    def _get_single_batch(self, pointer):
        batch = self.generate_batch(pointer)
        data = pd.DataFrame(batch)
        original_ends = data.v_sequence_end
        (
            v_start,
            v_end,
            d_start,
            d_end,
            j_start,
            j_end,
            padded_sequences,
        ) = self.process_sequences(data, corrupt_beginning=self.corrupt_beginning)

        removed = original_ends - (v_end - v_start)
        x = {
            "tokenized_sequence": padded_sequences
        }

        v_segment = []
        d_segment = []
        j_segment = []

        for s, e in zip(v_start, v_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            v_segment.append(empty)

        for s, e in zip(d_start, d_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            d_segment.append(empty)

        for s, e in zip(j_start, j_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            j_segment.append(empty)

        v_segment = np.vstack(v_segment)
        d_segment = np.vstack(d_segment)
        j_segment = np.vstack(j_segment)

        y = {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": self.get_expanded_ohe("V", data.v_allele, removed, original_ends),
            "d_allele": self.get_ohe("D", data.d_allele),
            "j_allele": self.get_ohe("J", data.j_allele),
        }
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
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "d_allele_count": self.d_allele_count,
            "j_allele_count": self.j_allele_count,
        }

    def tokenize_sequences(self, sequences, verbose=False):
        padded_sequences = []

        if verbose:
            iterator = tqdm(sequences, total=len(sequences))
        else:
            iterator = sequences

        for seq in iterator:
            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)
        padded_sequences = np.vstack(padded_sequences)

        return padded_sequences

    def __len__(self):
        return len(self.data)

class VDeepJDatasetSingleBeamSegmentationV1__5():
    """
    This support Short-D label + Mutation Rate head
    """
    def __init__(self, data_path, batch_size=64, max_sequence_length=512, batch_read_file=False, nrows=None,
                 mutation_rate=0.08, shm_flat=False,
                 randomize_rate=False,
                 corrupt_beginning=True, corrupt_proba=1, nucleotide_add_coef=35, nucleotide_remove_coef=50,
                 mutation_oracle_mode=False,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, allele_map_path='/home/bcrlab/thomas/AlignAIRR/',
                 seperator=','):
        self.max_sequence_length = max_sequence_length

        self.locus = global_genotype()
        self.max_seq_length = max_sequence_length
        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.add_remove_probability = st.bernoulli(0.5)
        self.corrupt_beginning = corrupt_beginning
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.mutation_oracle_mode = mutation_oracle_mode
        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP.pkl', 'rb') as h:
            self.VA = pickle.load(h)
            self.MAX_VA = max(self.VA[list(self.VA)[0]])

        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP_AT_END.pkl', 'rb') as h:
            self.VEND_SIM_MAP = pickle.load(h)
            self.MAX_VEND = max(self.VA[list(self.VA)[0]])

        self.sequence_corruptor = SequenceCorruptor(
            nucleotide_add_coef=nucleotide_add_coef, nucleotide_remove_coef=nucleotide_remove_coef, max_length=512,
            random_sequence_add_proba=random_sequence_add_proba, single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba,
            corrupt_proba=self.corrupt_proba,
            nucleotide_add_distribution=self.nucleotide_add_distribution,
            nucleotide_remove_distribution=self.nucleotide_remove_distribution

        )

        self.seperator = seperator
        self.batch_read_file = batch_read_file
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_allele',
                                      'd_allele', 'j_allele']

        self.mutate = True
        self.flat_vdj = True
        self.randomize_rate = randomize_rate
        self.no_trim_args = False
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.shm_flat = shm_flat
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()

        if not self.batch_read_file:
            self.data = pd.read_table(data_path,
                                      usecols=self.required_data_columns, nrows=nrows, sep=self.seperator)
            self.data_length = len(self.data)
        else:
            self.get_data_batch_generator(data_path)
            self.data_length = count_tsv_size(data_path)

    def get_data_batch_generator(self, path_to_data):
        self.data = table_generator(path_to_data, batch_size=self.batch_size, usecols=self.required_data_columns,
                                    seperator=self.seperator)

    def derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        j_alleles = sorted(list(self.j_dict))
        d_alleles = d_alleles + ['Short-D']

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
        }

    def derive_call_dictionaries(self):
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['V']}
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['D']}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['J']}

    def generate_batch(self, pointer):
        if not self.batch_read_file:
            data = {
                "sequence": [],
                "v_sequence_start": [],
                "v_sequence_end": [],
                "d_sequence_start": [],
                "d_sequence_end": [],
                "j_sequence_start": [],
                "j_sequence_end": [],
                "v_allele": [],
                "d_allele": [],
                "j_allele": [],
            }
            for idx, row in self.data.iloc[(pointer - self.batch_size):pointer, :].iterrows():
                data['sequence'].append(row['sequence'])
                data['v_sequence_start'].append(row['v_sequence_start'])
                data['v_sequence_end'].append(row['v_sequence_end'])
                data['d_sequence_start'].append(row['d_sequence_start'])
                data['d_sequence_end'].append(row['d_sequence_end'])
                data['j_sequence_start'].append(row['j_sequence_start'])
                data['j_sequence_end'].append(row['j_sequence_end'])
                data['v_allele'].append(row['v_allele'])
                data['d_allele'].append(row['d_allele'])
                data['j_allele'].append(row['j_allele'])

            return data
        else:
            read_batch = next(self.data)
            return read_batch

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
                start, end = (
                    whole_half_gap + 1,
                    self.max_seq_length - whole_half_gap - 1,
                )

        return trans_seq, start, end if iseven else (end + 1)

    def process_sequences(
            self, data: pd.DataFrame, corrupt_beginning=False, verbose=False
    ):
        return self.sequence_corruptor.process_sequences(data=data, corrupt_beginning=corrupt_beginning,
                                                         verbose=verbose)

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def get_ohe(self, type, values):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []
        for value in values:
            ohe = np.zeros(allele_count)
            ohe[allele_call_ohe[value]] = 1
            result.append(ohe)
        return np.vstack(result)

    def get_d_ohe(self,alleles,lengths):
        allele_count = self.properties_map['D']['allele_count']
        allele_call_ohe = self.properties_map['D']['allele_call_ohe']
        result = []
        for allele,length in zip(alleles,lengths):
            ohe = np.zeros(allele_count)
            if length >= 3:
                ohe[allele_call_ohe[allele]] = 1
            else:
                ohe[allele_call_ohe['Short-D']] = 1
            result.append(ohe)
        return np.vstack(result)
    def get_expanded_ohe(self, type, values, removed, ends_at=None):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []

        for value, remove, end in zip(values, removed, ends_at):
            ohe = np.zeros(allele_count)

            # get all alleles that are equally likely due missing nucleotides in the beginning
            equal_alleles = self.VA[value][min(remove, self.MAX_VA)]
            # get all allele that are equally likely due to missing nucleotides in the end
            equal_alleles += self.VEND_SIM_MAP[value][min(end, self.MAX_VEND)]

            for i in equal_alleles:
                ohe[allele_call_ohe[i]] = 1
            result.append(ohe)
        return np.vstack(result)

    def _get_single_batch(self, pointer):
        batch = self.generate_batch(pointer)
        data = pd.DataFrame(batch)
        original_ends = data.v_sequence_end
        (
            v_start,
            v_end,
            d_start,
            d_end,
            j_start,
            j_end,
            padded_sequences,
        ) = self.process_sequences(data, corrupt_beginning=self.corrupt_beginning)

        removed = original_ends - (v_end - v_start)
        d_length = (data.d_sequence_end - data.d_sequence_start).values
        x = {
            "tokenized_sequence": padded_sequences
        }

        v_segment = []
        d_segment = []
        j_segment = []

        for s, e in zip(v_start, v_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            v_segment.append(empty)

        for s, e in zip(d_start, d_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            d_segment.append(empty)

        for s, e in zip(j_start, j_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            j_segment.append(empty)

        v_segment = np.vstack(v_segment)
        d_segment = np.vstack(d_segment)
        j_segment = np.vstack(j_segment)

        y = {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": self.get_expanded_ohe("V", data.v_allele, removed, original_ends),
            "d_allele": self.get_d_ohe(data.d_allele,d_length),
            "j_allele": self.get_ohe("J", data.j_allele),
            'mutation_rate':data.mutation_rate.values
        }
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
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "d_allele_count": self.d_allele_count,
            "j_allele_count": self.j_allele_count,
        }

    def tokenize_sequences(self, sequences, verbose=False):
        padded_sequences = []

        if verbose:
            iterator = tqdm(sequences, total=len(sequences))
        else:
            iterator = sequences

        for seq in iterator:
            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)
        padded_sequences = np.vstack(padded_sequences)

        return padded_sequences

    def __len__(self):
        return len(self.data)

class VDeepJDatasetSingleBeamSegmentationV2():
    """
    In this version of the dataset we add insertions and deletions as well as the y variables
    need for those changes such as 3 deletions flags (for the classification nodes)
    a "short D" label to the D labels.
    and update the segmentation masks, so they ignore the insertion positions
    """

    def __init__(self, data_path, batch_size=64, max_sequence_length=512, batch_read_file=False, nrows=None,
                 mutation_rate=0.08, shm_flat=False,
                 randomize_rate=False,
                 corrupt_beginning=True, corrupt_proba=1, nucleotide_add_coef=35, nucleotide_remove_coef=50,
                 mutation_oracle_mode=False,
                 insertion_proba=0.5,
                 deletions_proba=0.5,
                 deletion_coef=10,
                 insertion_coef=10,
                 include_v_deletions = False,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, allele_map_path='/home/bcrlab/thomas/AlignAIRR/',
                 seperator=','):
        self.max_sequence_length = max_sequence_length

        self.locus = global_genotype()
        self.max_seq_length = max_sequence_length
        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.add_remove_probability = st.bernoulli(0.5)
        self.insertion_proba=insertion_proba
        self.deletions_proba=deletions_proba
        self.deletion_coef=deletion_coef
        self.insertion_coef=insertion_coef
        self.corrupt_beginning = corrupt_beginning
        self.corrupt_proba = corrupt_proba
        self.include_v_deletions = include_v_deletions
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.mutation_oracle_mode = mutation_oracle_mode
        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP.pkl', 'rb') as h:
            self.VA = pickle.load(h)
            self.MAX_VA = max(self.VA[list(self.VA)[0]])

        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP_AT_END.pkl', 'rb') as h:
            self.VEND_SIM_MAP = pickle.load(h)
            self.MAX_VEND = max(self.VA[list(self.VA)[0]])

        self.sequence_corruptor = SequenceCorruptorV2(
            nucleotide_add_coef=nucleotide_add_coef, nucleotide_remove_coef=nucleotide_remove_coef, max_length=512,
            random_sequence_add_proba=random_sequence_add_proba, single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba,
            corrupt_proba=self.corrupt_proba,
            insertion_proba=self.insertion_proba,
            deletions_proba=self.deletions_proba,
            deletion_coef=self.deletion_coef,
            insertion_coef=self.insertion_coef,
            nucleotide_add_distribution=self.nucleotide_add_distribution,
            nucleotide_remove_distribution=self.nucleotide_remove_distribution

        )

        self.seperator = seperator
        self.batch_read_file = batch_read_file
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_allele',
                                      'd_allele', 'j_allele']

        self.mutate = True
        self.flat_vdj = True
        self.randomize_rate = randomize_rate
        self.no_trim_args = False
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.shm_flat = shm_flat
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()

        if not self.batch_read_file:
            self.data = pd.read_table(data_path,
                                      usecols=self.required_data_columns, nrows=nrows, sep=self.seperator)
            self.data_length = len(self.data)
        else:
            self.get_data_batch_generator(data_path)
            self.data_length = count_tsv_size(data_path)

    def get_data_batch_generator(self, path_to_data):
        self.data = table_generator(path_to_data, batch_size=self.batch_size, usecols=self.required_data_columns,
                                    seperator=self.seperator)

    def derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        j_alleles = sorted(list(self.j_dict))
        d_alleles = d_alleles + ['Short-D']

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
        }

    def derive_call_dictionaries(self):
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['V']}
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['D']}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['J']}

    def generate_batch(self, pointer):
        if not self.batch_read_file:
            data = {
                "sequence": [],
                "v_sequence_start": [],
                "v_sequence_end": [],
                "d_sequence_start": [],
                "d_sequence_end": [],
                "j_sequence_start": [],
                "j_sequence_end": [],
                "v_allele": [],
                "d_allele": [],
                "j_allele": [],
            }
            for idx, row in self.data.iloc[(pointer - self.batch_size):pointer, :].iterrows():
                data['sequence'].append(row['sequence'])
                data['v_sequence_start'].append(row['v_sequence_start'])
                data['v_sequence_end'].append(row['v_sequence_end'])
                data['d_sequence_start'].append(row['d_sequence_start'])
                data['d_sequence_end'].append(row['d_sequence_end'])
                data['j_sequence_start'].append(row['j_sequence_start'])
                data['j_sequence_end'].append(row['j_sequence_end'])
                data['v_allele'].append(row['v_allele'])
                data['d_allele'].append(row['d_allele'])
                data['j_allele'].append(row['j_allele'])

            return data
        else:
            read_batch = next(self.data)
            return read_batch

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
                start, end = (
                    whole_half_gap + 1,
                    self.max_seq_length - whole_half_gap - 1,
                )

        return trans_seq, start, end if iseven else (end + 1)

    def process_sequences(
            self, data: pd.DataFrame, corrupt_beginning=False, verbose=False
    ):
        return self.sequence_corruptor.process_sequences(data=data, corrupt_beginning=corrupt_beginning,
                                                         verbose=verbose)

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def get_ohe(self, type, values):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []
        for value in values:
            ohe = np.zeros(allele_count)
            ohe[allele_call_ohe[value]] = 1
            result.append(ohe)
        return np.vstack(result)

    def get_d_ohe(self,alleles,lengths):
        allele_count = self.properties_map['D']['allele_count']
        allele_call_ohe = self.properties_map['D']['allele_call_ohe']
        result = []
        for allele,length in zip(alleles,lengths):
            ohe = np.zeros(allele_count)
            if length >= 3:
                ohe[allele_call_ohe[allele]] = 1
            else:
                ohe[allele_call_ohe['Short-D']] = 1
            result.append(ohe)
        return np.vstack(result)
    def get_expanded_ohe(self, type, values, removed, ends_at=None):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []

        for value, remove, end in zip(values, removed, ends_at):
            ohe = np.zeros(allele_count)

            # get all alleles that are equally likely due missing nucleotides in the beginning
            equal_alleles = self.VA[value][min(remove, self.MAX_VA)]
            # get all allele that are equally likely due to missing nucleotides in the end
            equal_alleles += self.VEND_SIM_MAP[value][min(end, self.MAX_VEND)]

            for i in equal_alleles:
                ohe[allele_call_ohe[i]] = 1
            result.append(ohe)
        return np.vstack(result)

    def _get_single_batch(self, pointer):
        batch = self.generate_batch(pointer)
        data = pd.DataFrame(batch)
        original_ends = data.v_sequence_end
        (
            v_start,
            v_end,
            d_start,
            d_end,
            j_start,
            j_end,
            padded_sequences,
            v_deletion,
            d_deletion,
            j_deletion,
            removed_by_event

        ) = self.process_sequences(data, corrupt_beginning=self.corrupt_beginning)

        removed = removed_by_event
        d_length = (data.d_sequence_end - data.d_sequence_start).values
        x = {
            "tokenized_sequence": padded_sequences
        }

        v_segment = []
        d_segment = []
        j_segment = []

        for s, e in zip(v_start, v_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            v_segment.append(empty)

        for s, e in zip(d_start, d_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            d_segment.append(empty)

        for s, e in zip(j_start, j_end):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            j_segment.append(empty)

        v_segment = np.vstack(v_segment)
        d_segment = np.vstack(d_segment)
        j_segment = np.vstack(j_segment)


        y = {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": self.get_expanded_ohe("V", data.v_allele, removed, original_ends),
            "d_allele": self.get_d_ohe(data.d_allele,d_length),
            "j_allele": self.get_ohe("J", data.j_allele),
            'mutation_rate':data.mutation_rate.values,
            'v_deletion':v_deletion,
            'j_deletion':j_deletion
        }
        if self.include_v_deletions:
            y['d_deletion'] = d_deletion

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
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "d_allele_count": self.d_allele_count,
            "j_allele_count": self.j_allele_count,
        }

    def tokenize_sequences(self, sequences, verbose=False):
        padded_sequences = []

        if verbose:
            iterator = tqdm(sequences, total=len(sequences))
        else:
            iterator = sequences

        for seq in iterator:
            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)
        padded_sequences = np.vstack(padded_sequences)

        return padded_sequences

    def __len__(self):
        return len(self.data)


class VDeepJDatasetSingleBeamSegmentationV2_t():
    """
    In this version of the dataset we add insertions and deletions as well as the y variables
    need for those changes such as 3 deletions flags (for the classification nodes)
    a "short D" label to the D labels.
    and update the segmentation masks, so they ignore the insertion positions
    """

    def __init__(self, data_path, batch_size=64, max_sequence_length=512, batch_read_file=False, nrows=None,
                 mutation_rate=0.08, shm_flat=False,
                 randomize_rate=False,
                 corrupt_beginning=True, corrupt_proba=1, nucleotide_add_coef=35, nucleotide_remove_coef=50,
                 insertion_proba=0.5,
                 deletions_proba=0.5,
                 deletion_coef=10,
                 insertion_coef=10,
                 mutation_oracle_mode=False,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, allele_map_path='/home/bcrlab/thomas/AlignAIRR/',
                 seperator=','):
        self.max_sequence_length = max_sequence_length

        self.insertion_proba = insertion_proba,
        self.deletions_proba = deletions_proba,
        self.deletion_coef = deletion_coef,
        self.insertion_coef = insertion_coef,
        self.locus = global_genotype()
        self.max_seq_length = max_sequence_length
        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.add_remove_probability = st.bernoulli(0.5)
        self.corrupt_beginning = corrupt_beginning
        self.corrupt_proba = corrupt_proba
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.mutation_oracle_mode = mutation_oracle_mode
        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP.pkl', 'rb') as h:
            self.VA = pickle.load(h)
            self.MAX_VA = max(self.VA[list(self.VA)[0]])

        with open(allele_map_path + 'V_ALLELE_SIMILARITY_MAP_AT_END.pkl', 'rb') as h:
            self.VEND_SIM_MAP = pickle.load(h)
            self.MAX_VEND = max(self.VA[list(self.VA)[0]])

        self.sequence_corruptor = SequenceCorruptorV2(
            nucleotide_add_coef=nucleotide_add_coef, nucleotide_remove_coef=nucleotide_remove_coef, max_length=512,
            random_sequence_add_proba=random_sequence_add_proba, single_base_stream_proba=single_base_stream_proba,
            duplicate_leading_proba=duplicate_leading_proba,
            random_allele_proba=random_allele_proba,
            corrupt_proba=self.corrupt_proba,
            insertion_proba=0.5,
            deletions_proba=0.5,
            deletion_coef=10,
            insertion_coef=10,
            nucleotide_add_distribution=self.nucleotide_add_distribution,
            nucleotide_remove_distribution=self.nucleotide_remove_distribution

        )

        self.seperator = seperator
        self.batch_read_file = batch_read_file
        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_allele',
                                      'd_allele', 'j_allele']

        self.mutate = True
        self.flat_vdj = True
        self.randomize_rate = randomize_rate
        self.no_trim_args = False
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.shm_flat = shm_flat
        self.derive_call_dictionaries()
        self.derive_call_one_hot_representation()

        if not self.batch_read_file:
            self.data = pd.read_table(data_path,
                                      usecols=self.required_data_columns, nrows=nrows, sep=self.seperator)
            self.data_length = len(self.data)
        else:
            self.get_data_batch_generator(data_path)
            self.data_length = count_tsv_size(data_path)

    def get_data_batch_generator(self, path_to_data):
        self.data = table_generator(path_to_data, batch_size=self.batch_size, usecols=self.required_data_columns,
                                    seperator=self.seperator)

    def derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        d_alleles = d_alleles+['Short-D'] # add short D label
        j_alleles = sorted(list(self.j_dict))

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
        }

    def derive_call_dictionaries(self):
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['V']}
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['D']}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['J']}

    def generate_batch(self, pointer):
        if not self.batch_read_file:
            data = {
                "sequence": [],
                "v_sequence_start": [],
                "v_sequence_end": [],
                "d_sequence_start": [],
                "d_sequence_end": [],
                "j_sequence_start": [],
                "j_sequence_end": [],
                "v_allele": [],
                "d_allele": [],
                "j_allele": [],
            }
            for idx, row in self.data.iloc[(pointer - self.batch_size):pointer, :].iterrows():
                data['sequence'].append(row['sequence'])
                data['v_sequence_start'].append(row['v_sequence_start'])
                data['v_sequence_end'].append(row['v_sequence_end'])
                data['d_sequence_start'].append(row['d_sequence_start'])
                data['d_sequence_end'].append(row['d_sequence_end'])
                data['j_sequence_start'].append(row['j_sequence_start'])
                data['j_sequence_end'].append(row['j_sequence_end'])
                data['v_allele'].append(row['v_allele'])
                data['d_allele'].append(row['d_allele'])
                data['j_allele'].append(row['j_allele'])

            return data
        else:
            read_batch = next(self.data)
            return read_batch

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
                start, end = (
                    whole_half_gap + 1,
                    self.max_seq_length - whole_half_gap - 1,
                )

        return trans_seq, start, end if iseven else (end + 1)

    def process_sequences(
            self, data: pd.DataFrame, corrupt_beginning=False, verbose=False
    ):
        return self.sequence_corruptor.process_sequences(data=data, corrupt_beginning=corrupt_beginning,
                                                         verbose=verbose)

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def get_ohe(self, type, values):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []
        for value in values:
            ohe = np.zeros(allele_count)
            ohe[allele_call_ohe[value]] = 1
            result.append(ohe)
        return np.vstack(result)

    def d_one_hot_encoding(self,d_length,d_allele):
        result = []
        allele_count = self.properties_map['D']['allele_count']
        allele_call_ohe = self.properties_map['D']['allele_call_ohe']
        for value,d_l in zip(d_allele,d_length):
            ohe = np.zeros(allele_count)
            if d_l >= 3:
                ohe[allele_call_ohe[value]] = 1
            else:
                ohe[allele_call_ohe['Short-D']] = 1

            result.append(ohe)
        return np.vstack(result)


    def get_expanded_ohe(self, type, values, removed, ends_at=None):
        allele_count = self.properties_map[type]['allele_count']
        allele_call_ohe = self.properties_map[type]['allele_call_ohe']
        result = []

        for value, remove, end in zip(values, removed, ends_at):
            ohe = np.zeros(allele_count)
            #remove = 0 if remove < 0 else remove # temporary fix!
            # get all alleles that are equally likely due missing nucleotides in the beginning
            equal_alleles = self.VA[value][min(remove, self.MAX_VA)]
            # get all allele that are equally likely due to missing nucleotides in the end
            equal_alleles += self.VEND_SIM_MAP[value][min(end, self.MAX_VEND)]

            for i in equal_alleles:
                ohe[allele_call_ohe[i]] = 1
            result.append(ohe)
        return np.vstack(result)

    def _get_single_batch(self, pointer):
        batch = self.generate_batch(pointer)
        data = pd.DataFrame(batch)
        original_ends = data.v_sequence_end
        (
            v_start,
            v_end,
            d_start,
            d_end,
            j_start,
            j_end,
            padded_sequences,
            deletion_flags,
            insertion_history,
            removed_from_v_start
        ) = self.process_sequences(data, corrupt_beginning=self.corrupt_beginning)

        removed = removed_from_v_start
        d_length = d_end-d_start

        x = {
            "tokenized_sequence": padded_sequences
        }

        v_segment = []
        d_segment = []
        j_segment = []


        for s, e,i_history in zip(v_start, v_end,insertion_history):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            if len(i_history['v']) > 0:
                for i in i_history['v']:
                    empty[0,i] = 0

            v_segment.append(empty)

        for s, e, i_history,d_l in zip(d_start, d_end,insertion_history,d_length):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            if len(i_history['d']) > 0:
                for i in i_history['d']:
                    empty[0,i] = 0
            d_segment.append(empty)

        for s, e, i_history in zip(j_start, j_end,insertion_history):
            empty = np.zeros((1, self.max_seq_length))
            empty[0, s:e] = 1
            if len(i_history['j']) > 0:
                for i in i_history['j']:
                    empty[0,i] = 0
            j_segment.append(empty)

        v_segment = np.vstack(v_segment)
        d_segment = np.vstack(d_segment)
        j_segment = np.vstack(j_segment)

        v_deletion = []
        d_deletion = []
        j_deletion = []

        for _sample in deletion_flags:
            v_deletion.append(int(_sample['v']))
            d_deletion.append(int(_sample['d']))
            j_deletion.append(int(_sample['j']))

        v_deletion = np.array(v_deletion)
        d_deletion = np.array(d_deletion)
        j_deletion = np.array(j_deletion)



        y = {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": self.get_expanded_ohe("V", data.v_allele, removed, original_ends),
            "d_allele": self.d_one_hot_encoding(d_length,data.d_allele),
            "j_allele": self.get_ohe("J", data.j_allele),
            'v_deletion':v_deletion.reshape(-1,1),
            'd_deletion':d_deletion.reshape(-1,1),
            'j_deletion':j_deletion.reshape(-1,1),
            'mutation_rate':data['mutation_rate'].values.reshape(-1,1)
        }
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
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "d_allele_count": self.d_allele_count,
            "j_allele_count": self.j_allele_count,
        }

    def tokenize_sequences(self, sequences, verbose=False):
        padded_sequences = []

        if verbose:
            iterator = tqdm(sequences, total=len(sequences))
        else:
            iterator = sequences

        for seq in iterator:
            padded_array, start, end = self._process_and_dpad(seq, self.max_seq_length)
            padded_sequences.append(padded_array)
        padded_sequences = np.vstack(padded_sequences)

        return padded_sequences

    def __len__(self):
        return len(self.data)
