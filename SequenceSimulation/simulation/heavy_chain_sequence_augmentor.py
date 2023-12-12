import pickle
import random
import numpy as np
import scipy.stats as st

from AlingAIRR_Sequence_Simulator import Event
from SequenceSimulation.sequence import HeavyChainSequence
from SequenceSimulation.simulation import SequenceAugmentorArguments
from SequenceSimulation.simulation.sequence_augmentor_base import SequenceAugmentorBase
from SequenceSimulation.utilities.data_config import DataConfig
import base64


class HeavyChainSequenceAugmentor(SequenceAugmentorBase):
    alleles_used = ['v','d','j']
    def __init__(self, dataconfig: DataConfig, args: SequenceAugmentorArguments = SequenceAugmentorArguments()):
        super().__init__(dataconfig,args)

        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.nucleotide_add_after_remove_distribution = st.beta(1, 3)

        self.short_d_length = args.short_d_length
        # Class Misc
        self.d_alleles = sorted([i for j in self.dataconfig.d_alleles for i in self.dataconfig.d_alleles[j]],
                                key=lambda x: x.name)
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.d_alleles}

        # Loading Routines
        self.load_correction_maps()

    # Loading Routines
    def load_correction_maps(self):
        from importlib import resources
        """
        This will load the V start and V end maps that tell us given the amount removed from the start or end
        of a given allele what option are equally likely
        :return:
        """
        with resources.path('SequenceSimulation.data', 'IGHV_ALLELE_5_PRIME_SIMILARITY_MAP.pkl') as data_path:
            with open(data_path, 'rb') as h:
                self.v_start_allele_correction_map = pickle.load(h)
                self.max_v_start_correction_map_value = max(
                    self.v_start_allele_correction_map[list(self.v_start_allele_correction_map)[0]])

        with resources.path('SequenceSimulation.data', 'IGHV_ALLELE_3_PRIME_SIMILARITY_MAP.pkl') as data_path:
            with open(data_path, 'rb') as h:
                self.v_end_allele_correction_map = pickle.load(h)
                self.max_v_end_correction_map_value = max(
                    self.v_end_allele_correction_map[list(self.v_end_allele_correction_map)[0]])

        with resources.path('SequenceSimulation.data', 'IGHD_TRIM_SIMILARITY_MAP.pkl') as data_path:
            with open(data_path, 'rb') as h:
                self.d_trim_correction_map = pickle.load(h)

    def correct_for_d_trims(self, simulated):
        # Get the 5' and 3' trims of the d allele in the simulated sequence
        trim_5 = simulated['d_trim_5']
        trim_3 = simulated['d_trim_3']
        # infer the precalculated map what alleles should be the ground truth for this sequence based on the trim
        simulated['d_allele'] = list(self.d_trim_correction_map[simulated['d_allele'][0]][(trim_5, trim_3)])

    def short_d_validation(self, simulated):
        d_length = simulated['d_sequence_end'] - simulated['d_sequence_start']
        if d_length < self.short_d_length:
            simulated['d_allele'] = ['Short-D']

    def fix_d_position_after_trimming_index_ambiguity(self, simulation):
        # Extract Current D Metadata
        d_start, d_end = simulation['d_sequence_start'], simulation['d_sequence_end']
        d_allele_remainder = simulation['sequence'][d_start:d_end]
        d_allele_ref = self.d_dict[simulation['d_allele'][0]]

        # Get the junction inserted after trimming to the sequence
        junction_5 = simulation['sequence'][simulation['v_sequence_end']:d_start]
        junction_3 = simulation['sequence'][d_end:simulation['j_sequence_start']]

        # Get the trimming lengths
        d_trim_5 = simulation['d_trim_5']
        d_trim_3 = simulation['d_trim_3']

        # Get the trimmed off sections from the reference
        trimmed_5 = d_allele_ref[:d_trim_5]
        trimmed_3 = d_allele_ref[len(d_allele_ref) - d_trim_3:]

        # check for overlap between generated junction and reference in the 5' trim
        for a, b in zip(trimmed_5[::-1], junction_5[::-1]):
            # in case the current poistion in the junction matches the reference exapnd the d segment
            if a == b:
                d_start -= 1
            else:  # if the continuous streak is broken or non-existent break!
                break

        # check for overlap between generated junction and reference in the 3' trim
        for a, b in zip(trimmed_3, junction_3):
            # in case the current poistion in the junction matches the reference exapnd the d segment
            if a == b:
                d_end += 1
            else:  # if the continious streak is broken or non existant break!
                break

        simulation['d_sequence_start'] = d_start
        simulation['d_sequence_end'] = d_end

    # Sequence Simulation
    def simulate_sequence(self):

        gen = HeavyChainSequence.create_random(self.dataconfig)
        gen.mutate(self.mutation_model)

        data = {
            "sequence": gen.mutated_seq,
            "v_sequence_start": gen.v_seq_start,
            "v_sequence_end": gen.v_seq_end,
            "d_sequence_start": gen.d_seq_start,
            "d_sequence_end": gen.d_seq_end,
            "j_sequence_start": gen.j_seq_start,
            "j_sequence_end": gen.j_seq_end,
            "v_allele": [gen.v_allele.name],
            "d_allele": [gen.d_allele.name],
            "j_allele": [gen.j_allele.name],
            'mutation_rate': gen.mutation_freq,
            'v_trim_5': gen.v_trim_5,
            'v_trim_3': gen.v_trim_3,
            'd_trim_5': gen.d_trim_5,
            'd_trim_3': gen.d_trim_3,
            'j_trim_5': gen.j_trim_5,
            'j_trim_3': gen.j_trim_3,
            'corruption_event': 'no-corruption',
            'corruption_add_amount': 0,
            'corruption_remove_amount': 0,
            'mutations': {pos: gen.mutations[pos] for pos in sorted(gen.mutations)},  # sort the mutations by position
            "Ns": dict()
        }
        return data

    def simulate_augmented_sequence(self):
        # 1. Simulate a Sequence from AIRRship
        simulated = self.simulate_sequence()

        # 1.1 Correction - Correct Start/End Positions Based on Generated Junctions
        self.fix_v_position_after_trimming_index_ambiguity(simulated)
        self.fix_d_position_after_trimming_index_ambiguity(simulated)
        self.fix_j_position_after_trimming_index_ambiguity(simulated)

        # 1.2 Correction - Add All V Alleles That Cant be Distinguished Based on the Amount Cut from the V Allele
        self.correct_for_v_end_cut(simulated)

        # 1.3 Correction - Add All D Alleles That Cant be Distinguished Based on the 5' and 3' Trims
        self.correct_for_d_trims(simulated)

        # 2. Corrupt Begging of Sequence ( V Start )
        if self.perform_corruption():
            # Inside this method, based on the corruption event we will also adjust the respective ground truth v
            # alleles
            self.corrupt_sequence_beginning(simulated)

        # 3. Add N's
        self.insert_Ns(simulated)

        # 4. Adjust D Allele , if the simulated length is smaller than short_d_length
        # class property, change the d allele to the "Short-D" label
        self.short_d_validation(simulated)

        self.process_before_return(simulated)

        return simulated
