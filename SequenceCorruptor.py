import importlib
import scipy.stats as st
import tensorflow as tf
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from airrship.create_repertoire import generate_sequence, load_data, get_genotype, create_allele_dict
from collections import defaultdict
import re
import random


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


class SequenceCorruptor:
    def __init__(self, nucleotide_add_coef=100, nucleotide_remove_coef=100, max_length=512,
                 random_sequence_add_proba=1,single_base_stream_proba=0,duplicate_leading_proba=0,
                 random_allele_proba=0,corrupt_proba=1):
        self.nucleotide_add_distribution = st.beta(1, 3)
        self.nucleotide_remove_distribution = st.beta(1, 3)
        self.corrupt_proba = corrupt_proba

        self.max_length = max_length
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.random_sequence_add_proba = random_sequence_add_proba
        self.single_base_stream_proba = single_base_stream_proba
        self.duplicate_leading_proba = duplicate_leading_proba
        self.random_allele_proba = random_allele_proba

        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }

        self.allele_library = global_genotype()
        self.V_alleles = self.allele_library[0]["V"]

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
        return np.random.choice([1,2,3],size=1,p = [0.4,0.4,0.2])

    def _sample_corruption_method(self):
        method = random.choices([self.random_nucleotides,self.single_base_stream,self.duplicate_leading,
                                 self.random_allele_section],
                       weights=[self.random_sequence_add_proba,self.single_base_stream_proba,
                                self.duplicate_leading_proba,self.random_allele_proba],k=1)[0]
        return method

    def _process_and_dpad(self, sequence, train=True):
        start, end = None, None
        trans_seq = [self.tokenizer_dictionary[i] for i in sequence]

        gap = self.max_length - len(trans_seq)
        iseven = gap % 2 == 0
        whole_half_gap = gap // 2

        if iseven:
            trans_seq = [0] * whole_half_gap + trans_seq + ([0] * whole_half_gap)
            if train:
                start, end = whole_half_gap, self.max_length - whole_half_gap - 1

        else:
            trans_seq = [0] * (whole_half_gap + 1) + trans_seq + ([0] * whole_half_gap)
            if train:
                start, end = (whole_half_gap + 1, self.max_length - whole_half_gap - 1)

        return trans_seq, start, end if iseven else (end + 1)

    def _fix_sequence_validity_after_corruption(self, sequence):
        if len(sequence) > self.max_length:
            # In case we added too many nucleotide to the beginning while corrupting the sequence, remove the slack
            slack = len(sequence) - self.max_length
            return sequence[slack:]
        else:
            return sequence

    def _corrupt_sequence_beginning(self, sequence):
        to_remove = self._sample_add_remove_distribution()
        if to_remove == 1: # remove
            amount_to_remove = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_remove:]
            return modified_sequence, True, amount_to_remove

        elif to_remove == 2: # add
            amount_to_add = self._sample_nucleotide_add_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_add,sequence)
            modified_sequence = self._fix_sequence_validity_after_corruption(
                modified_sequence
            )
            return modified_sequence, False, amount_to_add
        else: # add and remove
            #remove
            amount_to_removed = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_removed:]


            # add
            amount_to_added = self._sample_nucleotide_add_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_added,modified_sequence)
            modified_sequence = self._fix_sequence_validity_after_corruption(
                modified_sequence
            )
            return modified_sequence, True, amount_to_removed

    def _process_row(self, row, corrupt_beginning):
        to_corrupt = bool(np.random.binomial(1, self.corrupt_proba))
        if corrupt_beginning and to_corrupt:
            seq, was_removed, amount_changed = self._corrupt_sequence_beginning(row.sequence)
        else:
            seq = row.sequence

        padded_array, start, end = self._process_and_dpad(seq, self.max_length)

        if corrupt_beginning and to_corrupt:
            _adjust = start - amount_changed if was_removed else start + amount_changed
            start += amount_changed
        else:
            _adjust = start

        return (
            start,
            row.v_sequence_end + _adjust,
            row.d_sequence_start + _adjust,
            row.d_sequence_end + _adjust,
            row.j_sequence_start + _adjust,
            end,
            padded_array
        )

    def process_sequences(self, data: pd.DataFrame, corrupt_beginning=False, verbose=False):
        padded_sequences = []
        v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []

        data.loc[:,'to_corrupt'] = np.random.binomial(1, 0.4, size=len(data)).astype(bool)
        iterator = tqdm(data.itertuples(), total=len(data)) if verbose else data.itertuples()

        for row in iterator:
            if corrupt_beginning and row.to_corrupt:
                seq, was_removed, amount_changed = self._corrupt_sequence_beginning(
                    row.sequence
                )
            else:
                seq = row.sequence

            padded_array, start, end = self._process_and_dpad(seq, self.max_length)
            padded_sequences.append(padded_array)

            if corrupt_beginning and row.to_corrupt:
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

    def random_nucleotides(self, amount, sequence):
        random_seq = ''.join(random.choices(['A','T','C','G'],k=amount))
        return random_seq + sequence

    def duplicate_leading(self, amount, sequence):
        cap = amount if amount < len(sequence) else len(sequence)-1
        return sequence[:cap]+sequence

    def random_allele_section(self, amount, sequence):
        random_allele = random.choice(self.V_alleles).ungapped_seq.upper()
        cap = amount if amount < len(random_allele) else len(random_allele)-1
        return random_allele[:cap]+sequence

    def single_base_stream(self, amount, sequence):
        random_base = random.choice(['A','T','G','C','N'])*amount
        return random_base+sequence


