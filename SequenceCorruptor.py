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


# class SequenceCorruptor:
#     def __init__(self, nucleotide_add_coef=100, nucleotide_remove_coef=100, max_length=512,
#                  random_sequence_add_proba=1,single_base_stream_proba=0,duplicate_leading_proba=0,
#                  random_allele_proba=0,corrupt_proba=1,
#                  nucleotide_add_distribution=None,
#                  nucleotide_remove_distribution=None,
#                  ):


#         self.nucleotide_add_distribution = st.beta(2, 3) if nucleotide_add_distribution is None else nucleotide_add_distribution
#         self.nucleotide_remove_distribution = st.beta(2, 3) if nucleotide_remove_distribution is None else nucleotide_remove_distribution
#         self.corrupt_proba = corrupt_proba

#         self.max_length = max_length
#         self.nucleotide_add_coef = nucleotide_add_coef
#         self.nucleotide_remove_coef = nucleotide_remove_coef
#         self.random_sequence_add_proba = random_sequence_add_proba
#         self.single_base_stream_proba = single_base_stream_proba
#         self.duplicate_leading_proba = duplicate_leading_proba
#         self.random_allele_proba = random_allele_proba

#         self.tokenizer_dictionary = {
#             "A": 1,
#             "T": 2,
#             "G": 3,
#             "C": 4,
#             "N": 5,
#             "P": 0,  # pad token
#         }

#         self.allele_library = global_genotype()
#         self.V_alleles = self.allele_library[0]["V"]

#     def _sample_nucleotide_add_distribution(self, size):
#         sample = (
#                 self.nucleotide_add_coef * self.nucleotide_add_distribution.rvs(size=size)
#         ).astype(int)
#         if len(sample) == 1:
#             return sample.item()
#         else:
#             return sample

#     def _sample_nucleotide_remove_distribution(self, size):
#         sample = (
#                 self.nucleotide_remove_coef
#                 * self.nucleotide_remove_distribution.rvs(size=size)
#         ).astype(int)
#         if len(sample) == 1:
#             return sample.item()
#         else:
#             return sample

#     def _sample_add_remove_distribution(self):
#         return np.random.choice([1,2,3],size=1,p = [0.4,0.4,0.2])

#     def _sample_corruption_method(self):
#         method = random.choices([self.random_nucleotides,self.single_base_stream,self.duplicate_leading,
#                                  self.random_allele_section],
#                        weights=[self.random_sequence_add_proba,self.single_base_stream_proba,
#                                 self.duplicate_leading_proba,self.random_allele_proba],k=1)[0]
#         return method

#     def _process_and_dpad(self, sequence, train=True):
#         start, end = None, None
#         trans_seq = [self.tokenizer_dictionary[i] for i in sequence]

#         gap = self.max_length - len(trans_seq)
#         iseven = gap % 2 == 0
#         whole_half_gap = gap // 2

#         if iseven:
#             trans_seq = [0] * whole_half_gap + trans_seq + ([0] * whole_half_gap)
#             if train:
#                 start, end = whole_half_gap, self.max_length - whole_half_gap - 1

#         else:
#             trans_seq = [0] * (whole_half_gap + 1) + trans_seq + ([0] * whole_half_gap)
#             if train:
#                 start, end = (whole_half_gap + 1, self.max_length - whole_half_gap - 1)

#         return trans_seq, start, end if iseven else (end + 1)

#     def _fix_sequence_validity_after_corruption(self, sequence):
#         if len(sequence) > self.max_length:
#             # In case we added too many nucleotide to the beginning while corrupting the sequence, remove the slack
#             slack = len(sequence) - self.max_length
#             return sequence[slack:]
#         else:
#             return sequence

#     def _corrupt_sequence_beginning(self, sequence):
#         to_remove = self._sample_add_remove_distribution()
#         if to_remove == 1: # remove
#             amount_to_remove = self._sample_nucleotide_remove_distribution(1)
#             modified_sequence = sequence[amount_to_remove:]
#             return modified_sequence, True, amount_to_remove

#         elif to_remove == 2: # add
#             amount_to_add = self._sample_nucleotide_add_distribution(1)
#             method = self._sample_corruption_method()
#             modified_sequence = method(amount_to_add,sequence)
#             modified_sequence = self._fix_sequence_validity_after_corruption(
#                 modified_sequence
#             )
#             return modified_sequence, False, amount_to_add
#         else: # add and remove
#             #remove
#             amount_to_removed = self._sample_nucleotide_remove_distribution(1)
#             modified_sequence = sequence[amount_to_removed:]


#             # add
#             amount_to_added = self._sample_nucleotide_add_distribution(1)
#             method = self._sample_corruption_method()
#             modified_sequence = method(amount_to_added,modified_sequence)
#             modified_sequence = self._fix_sequence_validity_after_corruption(
#                 modified_sequence
#             )
#             return modified_sequence, False, amount_to_added-amount_to_removed

#     def _process_row(self, row, corrupt_beginning):
#         to_corrupt = bool(np.random.binomial(1, self.corrupt_proba))
#         if corrupt_beginning and to_corrupt:
#             seq, was_removed, amount_changed = self._corrupt_sequence_beginning(row.sequence)
#         else:
#             seq = row.sequence

#         padded_array, start, end = self._process_and_dpad(seq, self.max_length)

#         if corrupt_beginning and to_corrupt:
#             _adjust = start - amount_changed if was_removed else start + amount_changed
#             start += amount_changed
#         else:
#             _adjust = start

#         return (
#             start,
#             row.v_sequence_end + _adjust,
#             row.d_sequence_start + _adjust,
#             row.d_sequence_end + _adjust,
#             row.j_sequence_start + _adjust,
#             end,
#             padded_array
#         )

#     def process_sequences(self, data: pd.DataFrame, corrupt_beginning=False, verbose=False):
#         padded_sequences = []
#         v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []

#         data.loc[:,'to_corrupt'] = np.random.binomial(1, 0.4, size=len(data)).astype(bool)
#         iterator = tqdm(data.itertuples(), total=len(data)) if verbose else data.itertuples()

#         for row in iterator:
#             if corrupt_beginning and row.to_corrupt:
#                 seq, was_removed, amount_changed = self._corrupt_sequence_beginning(
#                     row.sequence
#                 )
#             else:
#                 seq = row.sequence

#             padded_array, start, end = self._process_and_dpad(seq, self.max_length)
#             padded_sequences.append(padded_array)

#             if corrupt_beginning and row.to_corrupt:
#                 if was_removed:
#                     # v is shorter
#                     _adjust = start-amount_changed
#                 else:
#                     # v is longer
#                     _adjust = start + amount_changed
#                     start += amount_changed
#             else:
#                 _adjust = start


#             v_start.append(start)
#             j_end.append(end)
#             v_end.append(row.v_sequence_end + _adjust)
#             d_start.append(row.d_sequence_start + _adjust)
#             d_end.append(row.d_sequence_end + _adjust)
#             j_start.append(row.j_sequence_start + _adjust)

#         v_start = np.array(v_start)
#         v_end = np.array(v_end)
#         d_start = np.array(d_start)
#         d_end = np.array(d_end)
#         j_start = np.array(j_start)
#         j_end = np.array(j_end)

#         padded_sequences = np.vstack(padded_sequences)

#         return v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences

#     def random_nucleotides(self, amount, sequence):
#         random_seq = ''.join(random.choices(['A','T','C','G'],k=amount))
#         return random_seq + sequence

#     def duplicate_leading(self, amount, sequence):
#         cap = amount if amount < len(sequence) else len(sequence)-1
#         return sequence[:cap]+sequence

#     def random_allele_section(self, amount, sequence):
#         random_allele = random.choice(self.V_alleles).ungapped_seq.upper()
#         cap = amount if amount < len(random_allele) else len(random_allele)-1
#         return random_allele[:cap]+sequence

#     def single_base_stream(self, amount, sequence):
#         random_base = random.choice(['A','T','G','C','N'])*amount
#         return random_base+sequence


class SequenceCorruptor:
    def __init__(self,
                 nucleotide_add_coef=100,
                 nucleotide_remove_coef=100,
                 nucleotide_add_remove_coef=50,
                 max_length=512,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, corrupt_proba=1,
                 nucleotide_add_distribution=None,
                 nucleotide_remove_distribution=None,
                 max_v_length=310,
                 ):

        self.nucleotide_add_distribution = st.beta(2,
                                                   3) if nucleotide_add_distribution is None else nucleotide_add_distribution
        self.nucleotide_remove_distribution = st.beta(2,
                                                      3) if nucleotide_remove_distribution is None else nucleotide_remove_distribution
        self.nucleotide_add_remove_distribution = st.beta(1, 3)
        self.corrupt_proba = corrupt_proba

        self.max_length = max_length
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.nucleotide_add_remove_coef = nucleotide_add_remove_coef

        self.random_sequence_add_proba = random_sequence_add_proba
        self.single_base_stream_proba = single_base_stream_proba
        self.duplicate_leading_proba = duplicate_leading_proba
        self.random_allele_proba = random_allele_proba
        self.max_v_length = max_v_length

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
            return min(sample.item(), self.max_v_length)
        else:
            return sample

    def _sample_nucleotide_add_remove_distribution(self, size):
        sample = (
                self.nucleotide_add_remove_coef
                * self.nucleotide_add_remove_distribution.rvs(size=size)
        ).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _sample_add_remove_distribution(self):
        return np.random.choice([1, 2, 3], size=1, p=[0.4, 0.4, 0.2])

    def _sample_corruption_method(self):
        method = random.choices([self.random_nucleotides, self.single_base_stream, self.duplicate_leading,
                                 self.random_allele_section],
                                weights=[self.random_sequence_add_proba, self.single_base_stream_proba,
                                         self.duplicate_leading_proba, self.random_allele_proba], k=1)[0]
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

    def _fix_sequence_validity_after_corruption(self, sequence, added):
        if len(sequence) > self.max_length:
            # In case we added too many nucleotide to the beginning while corrupting the sequence, remove the slack
            slack = len(sequence) - self.max_length
            return sequence[slack:], added - slack
        else:
            return sequence, added

    def _corrupt_sequence_beginning(self, sequence):
        action = self._sample_add_remove_distribution()

        if action == 1:  # remove
            amount_to_remove = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_remove:]
            return modified_sequence, 1, amount_to_remove

        elif action == 2:  # add
            amount_to_add = self._sample_nucleotide_add_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_add, sequence)
            modified_sequence, amount_to_add = self._fix_sequence_validity_after_corruption(
                modified_sequence, amount_to_add
            )
            return modified_sequence, 2, amount_to_add
        else:  # add and remove
            # remove
            amount_to_removed = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_removed:]

            # add
            amount_to_added = self._sample_nucleotide_add_remove_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_added, modified_sequence)
            modified_sequence, amount_to_added = self._fix_sequence_validity_after_corruption(
                modified_sequence, amount_to_added
            )
            return modified_sequence, 3, (amount_to_added, amount_to_removed)

    def process_sequences(self, data: pd.DataFrame, corrupt_beginning=False, verbose=False):
        padded_sequences = []
        v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []

        data.loc[:, 'to_corrupt'] = np.random.binomial(1, self.corrupt_proba, size=len(data)).astype(bool)
        iterator = tqdm(data.itertuples(), total=len(data)) if verbose else data.itertuples()

        for row in iterator:
            if corrupt_beginning and row.to_corrupt:
                seq, action, amount_changed = self._corrupt_sequence_beginning(
                    row.sequence
                )
            else:
                seq = row.sequence

            adjusted_v_start = None
            adjusted_v_end = None
            adjusted_d_start = None
            adjusted_d_end = None
            adjusted_j_start = None
            adjusted_j_end = None

            if corrupt_beginning and row.to_corrupt:
                if action == 1:  # remove only
                    amount_removed = amount_changed
                    adjusted_v_start = 0
                    adjusted_v_end = row.v_sequence_end - amount_removed
                    adjusted_d_start = row.d_sequence_start - amount_removed
                    adjusted_d_end = row.d_sequence_end - amount_removed
                    adjusted_j_start = row.j_sequence_start - amount_removed
                    adjusted_j_end = row.j_sequence_end - amount_removed

                elif action == 2:  # add only:
                    amount_added = amount_changed
                    adjusted_v_start = amount_added
                    adjusted_v_end = row.v_sequence_end + amount_added
                    adjusted_d_start = row.d_sequence_start + amount_added
                    adjusted_d_end = row.d_sequence_end + amount_added
                    adjusted_j_start = row.j_sequence_start + amount_added
                    adjusted_j_end = row.j_sequence_end + amount_added

                elif action == 3:  # remove and add
                    amount_added, amount_removed = amount_changed

                    adjusted_v_start = 0
                    adjusted_v_end = row.v_sequence_end - amount_removed
                    adjusted_d_start = row.d_sequence_start - amount_removed
                    adjusted_d_end = row.d_sequence_end - amount_removed
                    adjusted_j_start = row.j_sequence_start - amount_removed
                    adjusted_j_end = row.j_sequence_end - amount_removed

                    adjusted_v_start = adjusted_v_start + amount_added
                    adjusted_v_end = adjusted_v_end + amount_added
                    adjusted_d_start = adjusted_d_start + amount_added
                    adjusted_d_end = adjusted_d_end + amount_added
                    adjusted_j_start = adjusted_j_start + amount_added
                    adjusted_j_end = adjusted_j_end + amount_added
            else:
                adjusted_v_start = row.v_sequence_start
                adjusted_v_end = row.v_sequence_end
                adjusted_d_start = row.d_sequence_start
                adjusted_d_end = row.d_sequence_end
                adjusted_j_start = row.j_sequence_start
                adjusted_j_end = row.j_sequence_end

            padded_array, start, end = self._process_and_dpad(seq, self.max_length)
            pad_size = start
            padded_sequences.append(padded_array)

            adjusted_v_start += pad_size
            adjusted_v_end += pad_size
            adjusted_d_start += pad_size
            adjusted_d_end += pad_size
            adjusted_j_start += pad_size
            adjusted_j_end += pad_size

            v_start.append(adjusted_v_start)
            v_end.append(adjusted_v_end)
            d_start.append(adjusted_d_start)
            d_end.append(adjusted_d_end)
            j_start.append(adjusted_j_start)
            j_end.append(adjusted_j_end)

        v_start = np.array(v_start)
        v_end = np.array(v_end)
        d_start = np.array(d_start)
        d_end = np.array(d_end)
        j_start = np.array(j_start)
        j_end = np.array(j_end)

        padded_sequences = np.vstack(padded_sequences)

        return v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences

    def random_nucleotides(self, amount, sequence):
        random_seq = ''.join(random.choices(['A', 'T', 'C', 'G'], k=amount))
        return random_seq + sequence

    def duplicate_leading(self, amount, sequence):
        cap = amount if amount < len(sequence) else len(sequence) - 1
        return sequence[:cap] + sequence

    def random_allele_section(self, amount, sequence):
        random_allele = random.choice(self.V_alleles).ungapped_seq.upper()
        cap = amount if amount < len(random_allele) else len(random_allele) - 1
        return random_allele[:cap] + sequence

    def single_base_stream(self, amount, sequence):
        random_base = random.choice(['A', 'T', 'G', 'C', 'N']) * amount
        return random_base + sequence


class PositionContainer:
    def __init__(self, v_start, v_end, d_start, d_end, j_start, j_end):
        self.v_start = v_start
        self.v_end = v_end
        self.d_start = d_start
        self.d_end = d_end
        self.j_start = j_start
        self.j_end = j_end
        self.removed_from_v_start = 0

    def adjust_after_v_corruption(self, action, amount_changed):
        if action == 1:  # remove only
            amount_removed = amount_changed
            self.v_start = 0
            self.v_end = self.v_end - amount_removed
            self.d_start = self.d_start - amount_removed
            self.d_end = self.d_end - amount_removed
            self.j_start = self.j_start - amount_removed
            self.j_end = self.j_end - amount_removed
            self.removed_from_v_start = amount_removed

        elif action == 2:  # add only:
            amount_added = amount_changed
            self.v_start = amount_added
            self.v_end = self.v_end + amount_added
            self.d_start = self.d_start + amount_added
            self.d_end = self.d_end + amount_added
            self.j_start = self.j_start + amount_added
            self.j_end = self.j_end + amount_added

        elif action == 3:  # remove and add
            amount_added, amount_removed = amount_changed

            self.v_start = 0
            self.v_end = self.v_end - amount_removed
            self.d_start = self.d_start - amount_removed
            self.d_end = self.d_end - amount_removed
            self.j_start = self.j_start - amount_removed
            self.j_end = self.j_end - amount_removed

            self.v_start = self.v_start + amount_added
            self.v_end = self.v_end + amount_added
            self.d_start = self.d_start + amount_added
            self.d_end = self.d_end + amount_added
            self.j_start = self.j_start + amount_added
            self.j_end = self.j_end + amount_added

            self.removed_from_v_start = amount_removed

    def insert(self, position):
        if position <= self.v_end:
            self.v_end += 1
            self.d_start += 1
            self.d_end += 1
            self.j_start += 1
            self.j_end += 1
        elif position <= self.d_end:
            self.d_end += 1
            self.j_start += 1
            self.j_end += 1
        elif position <= self.j_end:
            self.j_end += 1

    def delete(self, position):
        # Check if the deletion is within the range of the container
        if position < self.v_start or position > self.j_end:
            raise ValueError("Deletion position is out of range")

        # If the deletion is within the 'v' segment
        if self.v_start <= position <= self.v_end:
            self.v_end -= 1
            self.d_start -= 1
            self.d_end -= 1
            self.j_start -= 1
            self.j_end -= 1

        # Else if the deletion is between 'v' and 'd' segments
        elif self.v_end < position < self.d_start:
            self.d_start -= 1
            self.d_end -= 1
            self.j_start -= 1
            self.j_end -= 1

        # Else if the deletion is within the 'd' segment
        elif self.d_start <= position <= self.d_end:
            self.d_end -= 1
            self.j_start -= 1
            self.j_end -= 1

        # Else if the deletion is between 'd' and 'j' segments
        elif self.d_end < position < self.j_start:
            self.j_start -= 1
            self.j_end -= 1

        # Else if the deletion is within the 'j' segment
        elif self.j_start <= position <= self.j_end:
            self.j_end -= 1

        # Ensure that the end position of a segment never becomes less than its start position
        if self.v_end < self.v_start:
            self.v_end = self.v_start
        if self.d_end < self.d_start:
            self.d_end = self.d_start
        if self.j_end < self.j_start:
            self.j_end = self.j_start

    def sample_position(self, ignore_d=False):
        available_segments = []

        if self.v_end >= self.v_start:
            available_segments.append('v')
        if not ignore_d and self.d_end >= self.d_start:
            available_segments.append('d')
        if self.j_end >= self.j_start:
            available_segments.append('j')

        if not available_segments:
            raise ValueError("All segments have been reduced to a size of 0")

        segment = random.choice(available_segments)

        if segment == 'v':
            return 'v', random.randint(0, self.v_end - self.v_start)
        elif segment == 'd':
            return 'd', random.randint(0, self.d_end - self.d_start)
        else:  # segment == 'j'
            return 'j', random.randint(0, self.j_end - self.j_start)

    def insert_nucleotide(self, segment, relative_position, sequence):
        if segment == 'v':
            absolute_position = self.v_start + relative_position
        elif segment == 'd':
            absolute_position = self.d_start + relative_position
        else:  # segment == 'j'
            absolute_position = self.j_start + relative_position

        nucleotide = random.choice(['A', 'T', 'C', 'G'])
        self.insert(absolute_position)
        return sequence[:absolute_position] + nucleotide + sequence[absolute_position:]

    def delete_nucleotide(self, segment, relative_position, sequence):
        if segment == 'v':
            absolute_position = self.v_start + relative_position
        elif segment == 'd':
            absolute_position = self.d_start + relative_position
        else:  # segment == 'j'
            absolute_position = self.j_start + relative_position

        self.delete(absolute_position)
        return sequence[:absolute_position] + sequence[absolute_position + 1:]

    def __str__(self):
        v_segment = f"V segment: Start {self.v_start}, End {self.v_end}"
        d_segment = f"D segment: Start {self.d_start}, End {self.d_end}"
        j_segment = f"J segment: Start {self.j_start}, End {self.j_end}"
        return "\n".join([v_segment, d_segment, j_segment])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return PositionContainer(self.v_start + other,
                                     self.v_end + other,
                                     self.d_start + other,
                                     self.d_end + other,
                                     self.j_start + other,
                                     self.j_end + other)


class SequenceCorruptorV2:
    """
    this version also added insertions and deletions
    """
    def __init__(self, nucleotide_add_coef=100, nucleotide_remove_coef=100, nucleotide_add_remove_coef=50,
                 max_length=512,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, corrupt_proba=1,
                 insertion_proba=0.5,
                 deletions_proba=0.5,
                 deletion_coef=10,
                 insertion_coef=10,
                 deletion_amount_distribution=None,
                 insertion_amount_distribution=None,
                 nucleotide_add_distribution=None,
                 nucleotide_remove_distribution=None,
                 max_v_length=310,
                 ):

        self.nucleotide_add_distribution = st.beta(2,
                                                   3) if nucleotide_add_distribution is None else nucleotide_add_distribution
        self.nucleotide_remove_distribution = st.beta(2,
                                                      3) if nucleotide_remove_distribution is None else nucleotide_remove_distribution
        self.nucleotide_add_remove_distribution = st.beta(1, 3)
        self.deletion_amount_distribution = st.beta(1,
                                                    3) if deletion_amount_distribution is None else deletion_amount_distribution
        self.insertion_amount_distribution = st.beta(1,
                                                     3) if insertion_amount_distribution is None else insertion_amount_distribution
        self.corrupt_proba = corrupt_proba

        self.deletion_coef = deletion_coef
        self.insertion_coef = insertion_coef

        self.max_length = max_length
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.nucleotide_add_remove_coef = nucleotide_add_remove_coef

        self.insertion_proba = insertion_proba
        self.deletions_proba = deletions_proba

        self.random_sequence_add_proba = random_sequence_add_proba
        self.single_base_stream_proba = single_base_stream_proba
        self.duplicate_leading_proba = duplicate_leading_proba
        self.random_allele_proba = random_allele_proba
        self.max_v_length = max_v_length

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

    def _deletion_amount_distribution(self, size):
        sample = 1 + (self.deletion_amount_distribution.rvs(size) * self.deletion_coef).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _insertion_amount_distribution(self, size):
        sample = 1 + (self.insertion_amount_distribution.rvs(size) * self.insertion_coef).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

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
            return min(sample.item(), self.max_v_length)
        else:
            return sample

    def _sample_nucleotide_add_remove_distribution(self, size):
        sample = (
                self.nucleotide_add_remove_coef
                * self.nucleotide_add_remove_distribution.rvs(size=size)
        ).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _sample_add_remove_distribution(self):
        return np.random.choice([1, 2, 3], size=1, p=[0.4, 0.4, 0.2])

    def _sample_corruption_method(self):
        method = random.choices([self.random_nucleotides, self.single_base_stream, self.duplicate_leading,
                                 self.random_allele_section],
                                weights=[self.random_sequence_add_proba, self.single_base_stream_proba,
                                         self.duplicate_leading_proba, self.random_allele_proba], k=1)[0]
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

    def _fix_sequence_validity_after_corruption(self, sequence, added):
        if len(sequence) > self.max_length:
            # In case we added too many nucleotide to the beginning while corrupting the sequence, remove the slack
            slack = len(sequence) - self.max_length
            return sequence[slack:], added - slack
        else:
            return sequence, added

    def _corrupt_sequence_beginning(self, sequence):
        action = self._sample_add_remove_distribution()

        if action == 1:  # remove
            amount_to_remove = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_remove:]
            return modified_sequence, 1, amount_to_remove

        elif action == 2:  # add
            amount_to_add = self._sample_nucleotide_add_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_add, sequence)
            modified_sequence, amount_to_add = self._fix_sequence_validity_after_corruption(
                modified_sequence, amount_to_add
            )
            return modified_sequence, 2, amount_to_add
        else:  # add and remove
            # remove
            amount_to_removed = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_removed:]

            # add
            amount_to_added = self._sample_nucleotide_add_remove_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_added, modified_sequence)
            modified_sequence, amount_to_added = self._fix_sequence_validity_after_corruption(
                modified_sequence, amount_to_added
            )
            return modified_sequence, 3, (amount_to_added, amount_to_removed)

    def process_sequences(self, data: pd.DataFrame, corrupt_beginning=False, verbose=False):
        padded_sequences = []
        v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []
        deletion_history = []
        removed_by_event = []

        data.loc[:, 'to_corrupt'] = np.random.binomial(1, self.corrupt_proba, size=len(data)).astype(bool)
        iterator = tqdm(data.itertuples(), total=len(data)) if verbose else data.itertuples()

        for row in iterator:
            d_length = row.d_sequence_end - row.d_sequence_start
            ignore_d = True if d_length < 3 else False
            position_container = PositionContainer(row.v_sequence_start, row.v_sequence_end,
                                                   row.d_sequence_start, row.d_sequence_end,
                                                   row.j_sequence_start, row.j_sequence_end)

            if corrupt_beginning and row.to_corrupt:
                seq, action, amount_changed = self._corrupt_sequence_beginning(
                    row.sequence
                )
                position_container.adjust_after_v_corruption(action=action, amount_changed=amount_changed)
            else:
                seq = row.sequence

            removed_by_event.append(row.v_sequence_end - (position_container.v_end - position_container.v_start))

            deletions = {'v': False, 'd': False, 'j': False}

            if np.random.binomial(1, self.deletions_proba) == 1:
                # make deletion
                amount_to_delete = self._deletion_amount_distribution(1)
                for i in range(amount_to_delete):
                    # sample a random position in one of the segments
                    random_segment, random_position = position_container.sample_position(ignore_d=ignore_d)
                    # update sequence
                    seq = position_container.delete_nucleotide(random_segment, random_position, seq)
                    # raise flag
                    deletions[random_segment] = True

            # Deletions

            padded_array, start, end = self._process_and_dpad(seq, self.max_length)
            pad_size = start
            padded_sequences.append(padded_array)

            position_container = position_container + pad_size

            deletion_history.append(deletions)

            v_start.append(position_container.v_start)
            v_end.append(position_container.v_end)
            d_start.append(position_container.d_start)
            d_end.append(position_container.d_end)
            j_start.append(position_container.j_start)
            j_end.append(position_container.j_end)

        v_start = np.array(v_start)
        v_end = np.array(v_end)
        d_start = np.array(d_start)
        d_end = np.array(d_end)
        j_start = np.array(j_start)
        j_end = np.array(j_end)

        padded_sequences = np.vstack(padded_sequences)

        v_deletion = []
        d_deletion = []
        j_deletion = []
        for record in deletion_history:
            v_deletion.append(record['v'])
            d_deletion.append(record['d'])
            j_deletion.append(record['j'])
        v_deletion = np.array(v_deletion).reshape(-1, 1)
        d_deletion = np.array(d_deletion).reshape(-1, 1)
        j_deletion = np.array(j_deletion).reshape(-1, 1)

        return v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences, \
               v_deletion, d_deletion, j_deletion, removed_by_event

    def random_nucleotides(self, amount, sequence):
        random_seq = ''.join(random.choices(['A', 'T', 'C', 'G'], k=amount))
        return random_seq + sequence

    def duplicate_leading(self, amount, sequence):
        cap = amount if amount < len(sequence) else len(sequence) - 1
        return sequence[:cap] + sequence

    def random_allele_section(self, amount, sequence):
        random_allele = random.choice(self.V_alleles).ungapped_seq.upper()
        cap = amount if amount < len(random_allele) else len(random_allele) - 1
        return random_allele[:cap] + sequence

    def single_base_stream(self, amount, sequence):
        random_base = random.choice(['A', 'T', 'G', 'C', 'N']) * amount
        return random_base + sequence


class SequenceCorruptorV2_t:
    """
    this version also added insertions and deletions
    """

    def __init__(self, nucleotide_add_coef=100, nucleotide_remove_coef=100, nucleotide_add_remove_coef=50,
                 max_length=512,
                 random_sequence_add_proba=1, single_base_stream_proba=0, duplicate_leading_proba=0,
                 random_allele_proba=0, corrupt_proba=1,
                 insertion_proba=0.5,
                 deletions_proba=0.5,
                 deletion_coef=10,
                 insertion_coef=10,
                 deletion_amount_distribution=None,
                 insertion_amount_distribution=None,
                 nucleotide_add_distribution=None,
                 nucleotide_remove_distribution=None,
                 max_v_length=310,
                 ):

        self.nucleotide_add_distribution = st.beta(2,
                                                   3) if nucleotide_add_distribution is None else nucleotide_add_distribution
        self.nucleotide_remove_distribution = st.beta(2,
                                                      3) if nucleotide_remove_distribution is None else nucleotide_remove_distribution
        self.nucleotide_add_remove_distribution = st.beta(1, 3)
        self.deletion_amount_distribution = st.beta(1,
                                                    3) if deletion_amount_distribution is None else deletion_amount_distribution
        self.insertion_amount_distribution = st.beta(1,
                                                     3) if insertion_amount_distribution is None else insertion_amount_distribution
        self.corrupt_proba = corrupt_proba

        self.deletion_coef = deletion_coef
        self.insertion_coef = insertion_coef

        self.max_length = max_length
        self.nucleotide_add_coef = nucleotide_add_coef
        self.nucleotide_remove_coef = nucleotide_remove_coef
        self.nucleotide_add_remove_coef = nucleotide_add_remove_coef

        self.insertion_proba = insertion_proba
        self.deletions_proba = deletions_proba

        self.random_sequence_add_proba = random_sequence_add_proba
        self.single_base_stream_proba = single_base_stream_proba
        self.duplicate_leading_proba = duplicate_leading_proba
        self.random_allele_proba = random_allele_proba
        self.max_v_length = max_v_length

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

    def _deletion_amount_distribution(self, size):
        sample = 1 + (self.deletion_amount_distribution.rvs(size) * self.deletion_coef).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _insertion_amount_distribution(self, size):
        sample = 1 + (self.insertion_amount_distribution.rvs(size) * self.insertion_coef).astype(int)
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
            return min(sample.item(), self.max_v_length)
        else:
            return sample

    def _sample_nucleotide_add_remove_distribution(self, size):
        sample = (
                self.nucleotide_add_remove_coef
                * self.nucleotide_add_remove_distribution.rvs(size=size)
        ).astype(int)
        if len(sample) == 1:
            return sample.item()
        else:
            return sample

    def _sample_add_remove_distribution(self):
        return np.random.choice([1, 2, 3], size=1, p=[0.4, 0.4, 0.2])

    def _sample_corruption_method(self):
        method = random.choices([self.random_nucleotides, self.single_base_stream, self.duplicate_leading,
                                 self.random_allele_section],
                                weights=[self.random_sequence_add_proba, self.single_base_stream_proba,
                                         self.duplicate_leading_proba, self.random_allele_proba], k=1)[0]
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

    def _fix_sequence_validity_after_corruption(self, sequence, added):
        if len(sequence) > self.max_length:
            # In case we added too many nucleotide to the beginning while corrupting the sequence, remove the slack
            slack = len(sequence) - self.max_length
            return sequence[slack:], added - slack
        else:
            return sequence, added

    def _corrupt_sequence_beginning(self, sequence):
        action = self._sample_add_remove_distribution()

        if action == 1:  # remove
            amount_to_remove = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_remove:]
            return modified_sequence, 1, amount_to_remove

        elif action == 2:  # add
            amount_to_add = self._sample_nucleotide_add_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_add, sequence)
            modified_sequence, amount_to_add = self._fix_sequence_validity_after_corruption(
                modified_sequence, amount_to_add
            )
            return modified_sequence, 2, amount_to_add
        else:  # add and remove
            # remove
            amount_to_removed = self._sample_nucleotide_remove_distribution(1)
            modified_sequence = sequence[amount_to_removed:]

            # add
            amount_to_added = self._sample_nucleotide_add_remove_distribution(1)
            method = self._sample_corruption_method()
            modified_sequence = method(amount_to_added, modified_sequence)
            modified_sequence, amount_to_added = self._fix_sequence_validity_after_corruption(
                modified_sequence, amount_to_added
            )
            return modified_sequence, 3, (amount_to_added, amount_to_removed)

    def get_amount_for_insertion(self, seq):
        amount_to_insert = self._insertion_amount_distribution(1)
        buffer = max([self.max_length - len(seq), 0])
        amount_to_insert = amount_to_insert if buffer > amount_to_insert else buffer
        return amount_to_insert

    def process_sequences(self, data: pd.DataFrame, corrupt_beginning=False, verbose=False):
        padded_sequences = []
        v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []
        deletion_flags = []
        insertion_history = []
        removed_from_v_start = []

        data.loc[:, 'to_corrupt'] = np.random.binomial(1, self.corrupt_proba, size=len(data)).astype(bool)
        iterator = tqdm(data.itertuples(), total=len(data)) if verbose else data.itertuples()

        for row in iterator:
            if corrupt_beginning and row.to_corrupt:
                seq, action, amount_changed = self._corrupt_sequence_beginning(
                    row.sequence
                )
            else:
                seq = row.sequence

            # sequence container for position operations
            position_container = PositionContainer(row.v_sequence_start, row.v_sequence_end,
                                                   row.d_sequence_start, row.d_sequence_end,
                                                   row.j_sequence_start, row.j_sequence_end)

            deletions = {'v': False, 'd': False, 'j': False}
            insertion_positions = {'v': [], 'd': [], 'j': []}
            d_length = row.d_sequence_end - row.d_sequence_start
            ignore_d = d_length < 3

            # V Start Corruption
            if corrupt_beginning and row.to_corrupt:
                position_container.adjust_after_v_corruption(action, amount_changed)

            # Insertions/Deletions
            # Insertion
            if np.random.binomial(1, self.insertion_proba) == 1:
                # make
                amount_to_insert = self.get_amount_for_insertion(seq)
                for i in range(amount_to_insert):
                    # sample a random position in one of the segments
                    random_segment, random_position = position_container.sample_position(ignore_d=ignore_d)
                    # update sequence
                    seq = position_container.insert_nucleotide(random_segment, random_position, seq)
                    # save position for mask adjustment
                    relative_start = {'v': position_container.v_start,
                                      'd': position_container.d_start,
                                      'j': position_container.j_start}

                    insertion_positions[random_segment].append(relative_start[random_segment] + random_position)

            if np.random.binomial(1, self.deletions_proba) == 1:
                # make deletion
                amount_to_delete = self._deletion_amount_distribution(1)
                for i in range(amount_to_delete):
                    # sample a random position in one of the segments
                    random_segment, random_position = position_container.sample_position(ignore_d=ignore_d)
                    # update sequence
                    seq = position_container.delete_nucleotide(random_segment, random_position, seq)
                    # raise flag
                    deletions[random_segment] = True

            padded_array, start, end = self._process_and_dpad(seq, self.max_length)
            pad_size = start
            padded_sequences.append(padded_array)

            position_container = position_container + pad_size
            for gene in insertion_positions:
                if len(insertion_positions[gene]) > 0:
                    insertion_positions[gene] = [i + pad_size for i in insertion_positions[gene]]

            v_start.append(position_container.v_start)
            v_end.append(position_container.v_end)
            d_start.append(position_container.d_start)
            d_end.append(position_container.d_end)
            j_start.append(position_container.j_start)
            j_end.append(position_container.j_end)
            insertion_history.append(insertion_positions)
            deletion_flags.append(deletions)
            removed_from_v_start.append(position_container.removed_from_v_start)

        v_start = np.array(v_start)
        v_end = np.array(v_end)
        d_start = np.array(d_start)
        d_end = np.array(d_end)
        j_start = np.array(j_start)
        j_end = np.array(j_end)

        padded_sequences = np.vstack(padded_sequences)

        return v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences, deletion_flags, insertion_history, \
               removed_from_v_start

    def random_nucleotides(self, amount, sequence):
        random_seq = ''.join(random.choices(['A', 'T', 'C', 'G'], k=amount))
        return random_seq + sequence

    def duplicate_leading(self, amount, sequence):
        cap = amount if amount < len(sequence) else len(sequence) - 1
        return sequence[:cap] + sequence

    def random_allele_section(self, amount, sequence):
        random_allele = random.choice(self.V_alleles).ungapped_seq.upper()
        cap = amount if amount < len(random_allele) else len(random_allele) - 1
        return random_allele[:cap] + sequence

    def single_base_stream(self, amount, sequence):
        random_base = random.choice(['A', 'T', 'G', 'C', 'N']) * amount
        return random_base + sequence
