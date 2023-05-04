import pickle
import re
from typing import Tuple, Dict, Union
import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm.auto import tqdm
import tensorflow as tf


class VDeepJDataPrepper:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.nucleotide_add_distribution = st.beta(1, 3)
        self.nucleotide_remove_distribution = st.beta(1, 3)
        self.add_remove_probability = st.bernoulli(0.5)
        self.batch_size = 64

        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }

    def process_sequences(
        self,
        data: pd.DataFrame,
        train: bool = True,
        corrupt_beginning=False,
        verbose=False,
    ):
        """
        This method takes in a pd.DataFrame of DNA sequences with their respective
        v/d/j start and end index positions (positions are only relevant for train) and returns the corresponding model inputs for either training or prediction
        based on the value of the `train` parameter. If `train` is True, the method returns four one-hot encoded numpy arrays
        representing the sequences for A, T, C, and G nucleotides. If `train` is False, the method also returns the starting and ending
        indices for the V, D, and J gene segments.

        :param sequences: A pd.DataFrame with the following columns: [sequence,v_sequence_end,d_sequence_start,
        d_sequence_end,j_sequence_start]. (start and end position columns are only relevant for train mode)
        :param train: A boolean parameter that determines whether the method should return inputs for training or prediction.
        :return: If `train` is False, returns a tuple containing four one-hot encoded numpy arrays for A, T, C, and G nucleotides:
            - a_oh_signal: A numpy array of one-hot encoded sequences for A nucleotides.
            - t_oh_signal: A numpy array of one-hot encoded sequences for T nucleotides.
            - c_oh_signal: A numpy array of one-hot encoded sequences for C nucleotides.
            - g_oh_signal: A numpy array of one-hot encoded sequences for G nucleotides.
        If `train` is True, returns a tuple containing the above four numpy arrays, plus the starting and ending indices
        for the V, D, and J gene segments:
            - v_start: The starting index for the V gene segment.
            - v_end: The ending index for the V gene segment.
            - d_start: The starting index for the D gene segment.
            - d_end: The ending index for the D gene segment.
            - j_start: The starting index for the J gene segment.
            - j_end: The ending index for the J gene segment.
        """

        padded_sequences = []
        # derive intervals only if train mode is True
        if train:
            v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []

        if verbose:
            iterator = tqdm(data.itertuples(), total=len(data))
        else:
            iterator = data.itertuples()

        for row in iterator:
            if train is True and corrupt_beginning is True:
                seq, was_removed, amount_changed = self._corrupt_sequence_beginning(
                    row.sequence
                )
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

            if train:
                v_start.append(start)
                j_end.append(end)
                v_end.append(row.v_sequence_end + _adjust)
                d_start.append(row.d_sequence_start + _adjust)
                d_end.append(row.d_sequence_end + _adjust)
                j_start.append(row.j_sequence_start + _adjust)

        if train:
            v_start = np.array(v_start)
            v_end = np.array(v_end)
            d_start = np.array(d_start)
            d_end = np.array(d_end)
            j_start = np.array(j_start)
            j_end = np.array(j_end)

        padded_sequences = np.vstack(padded_sequences)

        if train:
            return v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences
        else:
            return padded_sequences

    def _get_single_batch(
        self,
        data: pd.DataFrame,
        v_family_call_ohe_np,
        v_gene_call_ohe_np,
        v_allele_call_ohe_np,
        d_family_call_ohe_np,
        d_gene_call_ohe_np,
        d_allele_call_ohe_np,
        j_gene_call_ohe_np,
        j_allele_call_ohe_np,
        batch_size=64,
        train: bool = True,
        corrupt_beginning=False,
    ):
        if train:
            (
                v_start,
                v_end,
                d_start,
                d_end,
                j_start,
                j_end,
                padded_sequences,
            ) = self.process_sequences(data, train, corrupt_beginning)
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
                "v_family": v_family_call_ohe_np,
                "v_gene": v_gene_call_ohe_np,
                "v_allele": v_allele_call_ohe_np,
                "d_family": d_family_call_ohe_np,
                "d_gene": d_gene_call_ohe_np,
                "d_allele": d_allele_call_ohe_np,
                "j_gene": j_gene_call_ohe_np,
                "j_allele": j_allele_call_ohe_np,
            }
            return x, y
        else:
            padded_sequences = self.process_sequences(data, train, corrupt_beginning)
            x = {
                "tokenized_sequence": padded_sequences,
                "tokenized_sequence_for_masking": padded_sequences,
            }
            return x

    def _train_generator(
        self,
        data: pd.DataFrame,
        v_family_call_ohe_np,
        v_gene_call_ohe_np,
        v_allele_call_ohe_np,
        d_family_call_ohe_np,
        d_gene_call_ohe_np,
        d_allele_call_ohe_np,
        j_gene_call_ohe_np,
        j_allele_call_ohe_np,
        batch_size=64,
        train: bool = True,
        corrupt_beginning=False,
    ):
        self.batch_size = batch_size
        while True:
            num_examples = len(data)
            num_batches = num_examples // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size

                data_batch = data.iloc[start_idx:end_idx, :]
                (
                    v_start,
                    v_end,
                    d_start,
                    d_end,
                    j_start,
                    j_end,
                    padded_sequences,
                ) = self.process_sequences(data_batch, train, corrupt_beginning)

                batch_x = {
                    "tokenized_sequence": padded_sequences,
                    "tokenized_sequence_for_masking": padded_sequences,
                }
                batch_y = {
                    "v_start": v_start,
                    "v_end": v_end,
                    "d_start": d_start,
                    "d_end": d_end,
                    "j_start": j_start,
                    "j_end": j_end,
                    "v_family": v_family_call_ohe_np[start_idx:end_idx],
                    "v_gene": v_gene_call_ohe_np[start_idx:end_idx],
                    "v_allele": v_allele_call_ohe_np[start_idx:end_idx],
                    "d_family": d_family_call_ohe_np[start_idx:end_idx],
                    "d_gene": d_gene_call_ohe_np[start_idx:end_idx],
                    "d_allele": d_allele_call_ohe_np[start_idx:end_idx],
                    "j_gene": j_gene_call_ohe_np[start_idx:end_idx],
                    "j_allele": j_allele_call_ohe_np[start_idx:end_idx],
                }

                yield batch_x, batch_y

    def _eval_generator(
        self,
        data: pd.DataFrame,
        batch_size=64,
        train: bool = True,
        corrupt_beginning=False,
    ):
        self.batch_size = batch_size
        num_examples = len(data)
        num_batches = num_examples // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size

            data_batch = data.iloc[start_idx:end_idx, :]
            padded_sequences = self.process_sequences(
                data_batch, train, corrupt_beginning
            )

            batch_x = {
                "tokenized_sequence": padded_sequences,
                "tokenized_sequence_for_masking": padded_sequences,
            }

            yield batch_x

    def _get_tf_dataset_params(
        self,
        data: pd.DataFrame,
        v_family_call_ohe_np,
        v_gene_call_ohe_np,
        v_allele_call_ohe_np,
        d_family_call_ohe_np,
        d_gene_call_ohe_np,
        d_allele_call_ohe_np,
        j_gene_call_ohe_np,
        j_allele_call_ohe_np,
        batch_size=64,
        train: bool = True,
        corrupt_beginning=False,
    ):
        start_idx, end_idx = 0, 64
        x, y = self._get_single_batch(
            data[start_idx:end_idx],
            v_family_call_ohe_np[start_idx:end_idx],
            v_gene_call_ohe_np[start_idx:end_idx],
            v_allele_call_ohe_np[start_idx:end_idx],
            d_family_call_ohe_np[start_idx:end_idx],
            d_gene_call_ohe_np[start_idx:end_idx],
            d_allele_call_ohe_np[start_idx:end_idx],
            j_gene_call_ohe_np[start_idx:end_idx],
            j_allele_call_ohe_np[start_idx:end_idx],
            batch_size,
            train,
            corrupt_beginning,
        )

        output_types = ({k: tf.float32 for k in x}, {k: tf.float32 for k in y})

        output_shapes = (
            {k: (self.batch_size,) + x[k].shape[1:] for k in x},
            {k: (self.batch_size,) + y[k].shape[1:] for k in y},
        )

        return output_types, output_shapes

    def get_train_dataset(
        self,
        data: pd.DataFrame,
        v_family_call_ohe_np,
        v_gene_call_ohe_np,
        v_allele_call_ohe_np,
        d_family_call_ohe_np,
        d_gene_call_ohe_np,
        d_allele_call_ohe_np,
        j_gene_call_ohe_np,
        j_allele_call_ohe_np,
        batch_size=64,
        train: bool = True,
        corrupt_beginning=False,
    ):
        output_types, output_shapes = self._get_tf_dataset_params(
            data,
            v_family_call_ohe_np,
            v_gene_call_ohe_np,
            v_allele_call_ohe_np,
            d_family_call_ohe_np,
            d_gene_call_ohe_np,
            d_allele_call_ohe_np,
            j_gene_call_ohe_np,
            j_allele_call_ohe_np,
            batch_size,
            train,
            corrupt_beginning,
        )

        if train:
            dataset = tf.data.Dataset.from_generator(
                lambda: self._train_generator(
                    data,
                    v_family_call_ohe_np,
                    v_gene_call_ohe_np,
                    v_allele_call_ohe_np,
                    d_family_call_ohe_np,
                    d_gene_call_ohe_np,
                    d_allele_call_ohe_np,
                    j_gene_call_ohe_np,
                    j_allele_call_ohe_np,
                    batch_size,
                    train,
                    corrupt_beginning,
                ),
                output_types=output_types,
                output_shapes=output_shapes,
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: self._eval_generator(
                    data,
                    v_family_call_ohe_np,
                    v_gene_call_ohe_np,
                    v_allele_call_ohe_np,
                    d_family_call_ohe_np,
                    d_gene_call_ohe_np,
                    d_allele_call_ohe_np,
                    j_gene_call_ohe_np,
                    j_allele_call_ohe_np,
                    batch_size,
                    train,
                    corrupt_beginning,
                ),
                output_types=output_types,
                output_shapes=output_shapes,
            )

        return dataset

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

    def _convert_calls(self, column):
        """
        Private method, converts call series into 1 hot matrix and returns the mapping and the one hot matrix.
        :return:
        """
        call_ohe = pd.get_dummies(column)
        label_dict = {}
        for i, label in enumerate(call_ohe.columns):
            label_dict[label] = i
        call_ohe_np = call_ohe.to_numpy()
        return {en: i for en, i in enumerate(call_ohe.columns)}, call_ohe_np, label_dict

    def _process_and_dpad(self, sequence, train=False):
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

    def convert_calls_to_one_hot(self, gene, allele, family=None) -> Tuple:
        """
        This method takes in three Pandas series representing each of the V/D/J call types and converts them into
        one-hot encoded matrices. For each call level (family, gene, allele), this method returns a one-hot matrix
        and a dictionary mapping the one-hot position to the respective family/gene/allele.

        :param v_calls: Pandas series representing V call types.
        :param d_calls: Pandas series representing D call types.
        :param j_calls: Pandas series representing J call types.
        :return: Tuple containing:
            - One-hot encoded matrix and mapping dictionary for V family, gene, and allele call types.
            - One-hot encoded matrix and mapping dictionary for D family, gene, and allele call types.
            - One-hot encoded matrix and mapping dictionary for J gene and allele call types.
        """

        gene_call_ohe, gene_call_ohe_np, gene_label_dict = self._convert_calls(gene)
        allele_call_ohe, allele_call_ohe_np, allele_label_dict = self._convert_calls(
            allele
        )
        if family is not None:
            (
                family_call_ohe,
                family_call_ohe_np,
                family_label_dict,
            ) = self._convert_calls(family)
            return (
                family_label_dict,
                family_call_ohe,
                family_call_ohe_np,
                gene_label_dict,
                gene_call_ohe,
                gene_call_ohe_np,
                allele_label_dict,
                allele_call_ohe,
                allele_call_ohe_np,
            )
        else:
            return (
                gene_label_dict,
                gene_call_ohe,
                gene_call_ohe_np,
                allele_label_dict,
                allele_call_ohe,
                allele_call_ohe_np,
            )

    def get_family_gene_allele_map(
        self, v_calls=None, d_calls=None, j_calls=None
    ) -> Dict:
        """
        This method takes in three lists representing each of the V/D/J calls in the training sequence and creates
        a dictionary for each call, decomposing it into family, gene, and allele. The resulting dictionary for each
        call type (V/D/J) has the following format: {'family': family_list, 'gene': gene_list, 'allele': allele_list}.

        :param v_calls: A list of "V" calls in the training sequence.
        :param d_calls: A list of "D" calls in the training sequence.
        :param j_calls: A list of "J" calls in the training sequence.
        :return: A dictionary containing the decomposed information for each call type:
            - v_dict: Dictionary containing decomposed information for V calls, with the format {'family': family_list,
              'gene': gene_list, 'allele': allele_list}.
            - d_dict: Dictionary containing decomposed information for D calls, with the format {'family': family_list,
              'gene': gene_list, 'allele': allele_list}.
            - j_dict: Dictionary containing decomposed information for J calls, with the format {'family': family_list,
              'gene': gene_list, 'allele': allele_list}.
        """
        v_dict, d_dict, j_dict = None, None, None
        if v_calls is not None:
            v_dict = dict()
            for i in v_calls:
                fam, gene, allele = re.findall(
                    r"IGHV[A-ZA-Za-z0-9/]*|[0-9A-Za-z]+-?[0-9A-Za-z_]*", i
                )
                v_dict[i] = {"family": fam, "gene": gene, "allele": allele}
        if d_calls is not None:
            d_dict = dict()
            for i in d_calls:
                fam, gene, allele = re.findall(
                    r"IGHD[A-ZA-Za-z0-9/]*|[0-9A-Za-z]+-?[0-9A-Za-z]*", i
                )
                d_dict[i] = {"family": fam, "gene": gene, "allele": allele}
        if j_calls is not None:
            j_dict = dict()
            for i in j_calls:
                gene, allele = re.findall(
                    r"IGHJ[A-ZA-Za-z0-9/]*|[0-9A-Za-z]+-?[0-9A-Za-z]*", i
                )
                j_dict[i] = {"gene": gene, "allele": allele}

        return v_dict, d_dict, j_dict

    def create_sub_classes_dict(self, v_dict, d_dict, j_dict):
        def nested_get_sub_callses(d, v_d_or_j):
            sub_cllases = {}
            # V or D
            if v_d_or_j in ["v", "d"]:
                for call in d.values():
                    fam, gene, allele = call["family"], call["gene"], call["allele"]
                    if fam not in sub_cllases.keys():
                        sub_cllases[fam] = {}
                        sub_cllases[fam][gene] = {allele: allele}
                    else:
                        if gene not in sub_cllases[fam].keys():
                            sub_cllases[fam][gene] = {allele: allele}
                        else:
                            sub_cllases[fam][gene][allele] = allele
            # J
            elif v_d_or_j == "j":
                for call in d.values():
                    gene, allele = call["gene"], call["allele"]
                    if gene not in sub_cllases.keys():
                        sub_cllases[gene] = {allele: allele}
                    else:
                        sub_cllases[gene][allele] = allele
            return sub_cllases

        sub_cllases_v = nested_get_sub_callses(v_dict, "v")
        sub_cllases_d = nested_get_sub_callses(d_dict, "d")
        sub_cllases_j = nested_get_sub_callses(j_dict, "j")

        sub_cllases = {"v": sub_cllases_v, "d": sub_cllases_d, "j": sub_cllases_j}

        return sub_cllases
