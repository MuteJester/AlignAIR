import importlib

import scipy.stats as st
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from airrship.create_repertoire import generate_sequence,load_data,get_genotype,create_allele_dict
from collections import defaultdict
import re


def global_genotype():
    try:
        path_to_data = importlib.resources.files(
            'airrship').joinpath("data")
    except AttributeError:
        with importlib.resources.path('airrship', 'data') as p:
            path_to_data = p
    v_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHV.fasta")
    d_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHD.fasta")
    j_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHJ.fasta")

    vdj_allele_dicts = {"V": v_alleles,
                        "D": d_alleles,
                        "J": j_alleles}

    chromosome1, chromosome2 = defaultdict(list), defaultdict(list)
    for segment in ["V", "D", "J"]:
        allele_dict = vdj_allele_dicts[segment]
        for gene in allele_dict.values():
            for allele in gene:
                chromosome1[segment].append(allele)
                chromosome2[segment].append(allele)

    locus = [chromosome1, chromosome2]
    return locus


def translate_mutation_output(shm_events,max_seq_length):
    shm_vec = np.zeros(max_seq_length, dtype=np.uint8)
    pos = np.array(list(shm_events.keys()),dtype=np.uint8)
    pos -=1
    shm_vec[pos] = 1
    return shm_vec


class VDeepJUnbondedDataset():
    def __init__(self, batch_size=64, max_sequence_length=512, mutation_rate=0.08, shm_flat=False,randomize_rate=False,
                corrupt_beginning = True,corrupt_proba = 1,nucleotide_add_coef = 35,nucleotide_remove_coef=50,mutation_oracle_mode=False
                 ):
        self.max_sequence_length = max_sequence_length

        self.data_dict = load_data()
        self.locus = global_genotype()
        self.max_seq_length = max_sequence_length
        self.nucleotide_add_distribution = st.beta(1, 3)
        self.nucleotide_remove_distribution = st.beta(1, 3)
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

        self.mutate = True
        self.flat_vdj = True
        self.randomize_rate = randomize_rate
        self.no_trim_args = False
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.shm_flat = shm_flat

        self.derive_call_dictionaries()
        self.derive_counts()
        self.derive_call_one_hot_representation()
        self.derive_sub_classes_dict()

    def generate_single(self):
        if self.randomize_rate:
            return generate_sequence(self.locus, self.data_dict, mutate=self.mutate,
                                     mutation_rate=np.random.uniform(0,self.mutation_rate,1).item(),
                                     shm_flat=self.shm_flat, flat_usage='gene')
        else:
            return generate_sequence(self.locus, self.data_dict, mutate=self.mutate, mutation_rate=self.mutation_rate,
                                     shm_flat=self.shm_flat, flat_usage='gene')

    def derive_counts(self):
        self.v_family_count = len(set([self.v_dict[i]["family"] for i in self.v_dict]))
        self.v_gene_count = len(set([self.v_dict[i]["gene"] for i in self.v_dict]))
        self.v_allele_count = len(set([self.v_dict[i]["allele"] for i in self.v_dict]))

        self.d_family_count = len(set([self.d_dict[i]["family"] for i in self.d_dict]))
        self.d_gene_count = len(set([self.d_dict[i]["gene"] for i in self.d_dict]))
        self.d_allele_count = len(set([self.d_dict[i]["allele"] for i in self.d_dict]))

        self.j_gene_count = len(set([self.j_dict[i]["gene"] for i in self.j_dict]))
        self.j_allele_count = len(set([self.j_dict[i]["allele"] for i in self.j_dict]))

    def derive_call_one_hot_representation(self):
        v_families = sorted(set([self.v_dict[i]["family"] for i in self.v_dict]))
        v_genes = sorted(set([self.v_dict[i]["gene"] for i in self.v_dict]))
        v_alleles = sorted(set([self.v_dict[i]["allele"] for i in self.v_dict]))

        self.v_family_call_ohe = {f: i for i, f in enumerate(v_families)}
        self.v_gene_call_ohe = {f: i for i, f in enumerate(v_genes)}
        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}

        d_families = sorted(set([self.d_dict[i]["family"] for i in self.d_dict]))
        d_genes = sorted(set([self.d_dict[i]["gene"] for i in self.d_dict]))
        d_alleles = sorted(set([self.d_dict[i]["allele"] for i in self.d_dict]))

        self.d_family_call_ohe = {f: i for i, f in enumerate(d_families)}
        self.d_gene_call_ohe = {f: i for i, f in enumerate(d_genes)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}

        j_genes = sorted(set([self.j_dict[i]["gene"] for i in self.j_dict]))
        j_alleles = sorted(set([self.j_dict[i]["allele"] for i in self.j_dict]))

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

    def derive_call_dictionaries(self):
        self.v_dict, self.d_dict, self.j_dict = dict(), dict(), dict()
        for call in ["V", "D", "J"]:
            for idx in range(2):
                for N in self.locus[idx][call]:
                    if call == "V":
                        family, G = N.name.split("-", 1)
                        gene, allele = G.split("*")
                        self.v_dict[N.name] = {
                            "family": family,
                            "gene": gene,
                            "allele": allele,
                        }
                    elif call == "D":
                        family, G = N.name.split("-", 1)
                        gene, allele = G.split("*")
                        self.d_dict[N.name] = {
                            "family": family,
                            "gene": gene,
                            "allele": allele,
                        }
                    elif call == "J":
                        gene, allele = N.name.split("*")
                        self.j_dict[N.name] = {"gene": gene, "allele": allele}

    def decompose_call(self, type, call):
        if type == "V" or type == "D":
            family, G = call.split("-", 1)
            gene, allele = G.split("*")
            return family, gene, allele
        else:
            return call.split("*")

    def generate_batch(self):
        data = {
            "sequence": [],
            "v_sequence_start": [],
            "v_sequence_end": [],
            "d_sequence_start": [],
            "d_sequence_end": [],
            "j_sequence_start": [],
            "j_sequence_end": [],
            "v_family": [],
            "v_gene": [],
            "v_allele": [],
            "d_family": [],
            "d_gene": [],
            "d_allele": [],
            "j_gene": [],
            "j_allele": [],
        }
        if self.mutation_oracle_mode:
            data["mutations"] = []

        for _ in range(self.batch_size):
            gen = self.generate_single()
            v_family, v_gene, v_allele = self.decompose_call("V", gen.v_allele.name)
            d_family, d_gene, d_allele = self.decompose_call("D", gen.d_allele.name)
            j_gene, j_allele = self.decompose_call("J", gen.j_allele.name)

            data["sequence"].append(gen.mutated_seq)
            data["v_sequence_start"].append(gen.v_seq_start)
            data["v_sequence_end"].append(gen.v_seq_end)
            data["d_sequence_start"].append(gen.d_seq_start)
            data["d_sequence_end"].append(gen.d_seq_end)
            data["j_sequence_start"].append(gen.j_seq_start)
            data["j_sequence_end"].append(gen.j_seq_end)
            data["v_family"].append(v_family)
            data["v_gene"].append(v_gene)
            data["v_allele"].append(v_allele)
            data["d_family"].append(d_family)
            data["d_gene"].append(d_gene)
            data["d_allele"].append(d_allele)
            data["j_gene"].append(j_gene)
            data["j_allele"].append(j_allele)
            if self.mutation_oracle_mode:
                data["mutations"].append(translate_mutation_output(gen.mutations, 512))

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
                start, end = (
                    whole_half_gap + 1,
                    self.max_seq_length - whole_half_gap - 1,
                )

        return trans_seq, start, end if iseven else (end + 1)

    def process_sequences(
        self, data: pd.DataFrame, corrupt_beginning=False, verbose=False
    ):
        padded_sequences = []
        v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []

        if verbose:
            iterator = tqdm(data.itertuples(), total=len(data))
        else:
            iterator = data.itertuples()

        for row in iterator:
            to_corrupt = bool(np.random.binomial(1, self.corrupt_proba))
            if corrupt_beginning and to_corrupt:
                seq, was_removed, amount_changed = self._corrupt_sequence_beginning(
                    row.sequence
                )
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

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_family": get_reverse_dict(self.v_family_call_ohe),
            "v_gene": get_reverse_dict(self.v_gene_call_ohe),
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_family": get_reverse_dict(self.d_family_call_ohe),
            "d_gene": get_reverse_dict(self.d_gene_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_gene": get_reverse_dict(self.j_gene_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def get_ohe(self, type, level, values):
        if type == "V":
            if level == "family":
                result = []
                for value in values:
                    ohe = np.zeros(self.v_family_count)
                    ohe[self.v_family_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
            elif level == "gene":
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
        elif type == "D":
            if level == "family":
                result = []
                for value in values:
                    ohe = np.zeros(self.d_family_count)
                    ohe[self.d_family_call_ohe[value]] = 1
                    result.append(ohe)
                return np.vstack(result)
            elif level == "gene":
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
            if level == "gene":
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
        batch = self.generate_batch()
        if self.mutation_oracle_mode:
            mutations = batch.pop("mutations")
            mutations = np.vstack(mutations)
        data = pd.DataFrame(batch)
        (
            v_start,
            v_end,
            d_start,
            d_end,
            j_start,
            j_end,
            padded_sequences,
        ) = self.process_sequences(data, corrupt_beginning=self.corrupt_beginning)
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
            "v_family": self.get_ohe("V", "family", data.v_family),
            "v_gene": self.get_ohe("V", "gene", data.v_gene),
            "v_allele": self.get_ohe("V", "allele", data.v_allele),
            "d_family": self.get_ohe("D", "family", data.d_family),
            "d_gene": self.get_ohe("D", "gene", data.d_gene),
            "d_allele": self.get_ohe("D", "allele", data.d_allele),
            "j_gene": self.get_ohe("J", "gene", data.j_gene),
            "j_allele": self.get_ohe("J", "allele", data.j_allele),
        }

        if self.mutation_oracle_mode:
            y["mutations"] = mutations
        return x, y

    def _get_tf_dataset_params(self):
        x, y = self._get_single_batch()

        output_types = ({k: tf.float32 for k in x}, {k: tf.float32 for k in y})

        output_shapes = (
            {k: (self.batch_size,) + x[k].shape[1:] for k in x},
            {k: (self.batch_size,) + y[k].shape[1:] for k in y},
        )

        return output_types, output_shapes

    def _generate_random_nucleotide_sequence(self, length):
        sequence = "".join(np.random.choice(["A", "T", "C", "G"], size=length))
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
            modified_sequence = self._fix_sequence_validity_after_corruption(
                modified_sequence
            )
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
            output_shapes=output_shapes,
        )

        return dataset

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

    def _sample_nucleotide_add_distribution(self, size):
        sample = (
            self.nucleotide_add_coef * self.nucleotide_add_distribution.rvs(size=size)
        ).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _sample_nucleotide_remove_distribution(self, size):
        sample = (
            self.nucleotide_remove_coef
            * self.nucleotide_remove_distribution.rvs(size=size)
        ).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _sample_add_remove_distribution(self):
        return bool(self.add_remove_probability.rvs(1))

    def derive_sub_classes_dict(self):
        self.label_name_sub_classes_dict = self.data_prepper.load_sub_classes_dict()

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
                for fam in self.label_name_sub_classes_dict[v_d_or_j].keys():
                    label_num = self.label_num_sub_classes_dict[v_d_or_j]["family"][fam]
                    wanted_keys = list(
                        self.label_name_sub_classes_dict[v_d_or_j][fam].keys()
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
                for fam, fam_dict in self.label_name_sub_classes_dict[v_d_or_j].items():
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
                                self.label_name_sub_classes_dict[v_d_or_j][fam][
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
                for gene in self.label_name_sub_classes_dict[v_d_or_j].keys():
                    if gene not in self.ohe_sub_classes_dict[v_d_or_j]["gene"].keys():
                        label_num = self.label_num_sub_classes_dict[v_d_or_j]["gene"][
                            gene
                        ]
                        wanted_keys = list(
                            self.label_name_sub_classes_dict[v_d_or_j][gene].keys()
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

        # s = 4
        if 1:
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

    def create_vector(self, idxs, length):
        vector = np.zeros(length)
        vector[idxs] = 1
        return vector

    def __len__(self):
        return len(self.data)
