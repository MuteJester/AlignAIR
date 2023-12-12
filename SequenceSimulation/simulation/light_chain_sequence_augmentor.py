import pickle
import random

import numpy as np
import scipy.stats as st
from SequenceSimulation.sequence import LightChainType
from SequenceSimulation.sequence import LightChainSequence
from SequenceSimulation.simulation import SequenceAugmentorArguments
from SequenceSimulation.simulation.sequence_augmentor_base import SequenceAugmentorBase
from SequenceSimulation.utilities.data_config import DataConfig


class LightChainSequenceAugmentor(SequenceAugmentorBase):
    alleles_used = ['v', 'j']

    def __init__(self, dataconfig: DataConfig, args: SequenceAugmentorArguments = SequenceAugmentorArguments()):
        super().__init__(dataconfig, args)

        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.nucleotide_add_after_remove_distribution = st.beta(1, 3)
        self.chain_type = LightChainType.KAPPA if 'IGKV' in list(self.dataconfig.v_alleles)[
            0] else LightChainType.LAMBDA
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

        # for kappa
        if self.chain_type == LightChainType.KAPPA:
            with resources.path('SequenceSimulation.data', 'IGKV_ALLELE_5_PRIME_SIMILARITY_MAP.pkl') as data_path:
                with open(data_path, 'rb') as h:
                    self.v_start_allele_correction_map = pickle.load(h)
                    self.max_v_start_correction_map_value = max(
                        self.v_start_allele_correction_map[list(self.v_start_allele_correction_map)[0]])

            with resources.path('SequenceSimulation.data', 'IGKV_ALLELE_3_PRIME_SIMILARITY_MAP.pkl') as data_path:
                with open(data_path, 'rb') as h:
                    self.v_end_allele_correction_map = pickle.load(h)
                    self.max_v_end_correction_map_value = max(
                        self.v_end_allele_correction_map[list(self.v_end_allele_correction_map)[0]])
        else:
            # for lambda
            with resources.path('SequenceSimulation.data', 'IGLV_ALLELE_5_PRIME_SIMILARITY_MAP.pkl') as data_path:
                with open(data_path, 'rb') as h:
                    self.v_start_allele_correction_map = pickle.load(h)
                    self.max_v_start_correction_map_value = max(
                        self.v_start_allele_correction_map[list(self.v_start_allele_correction_map)[0]])

            with resources.path('SequenceSimulation.data', 'IGLV_ALLELE_3_PRIME_SIMILARITY_MAP.pkl') as data_path:
                with open(data_path, 'rb') as h:
                    self.v_end_allele_correction_map = pickle.load(h)
                    self.max_v_end_correction_map_value = max(
                        self.v_end_allele_correction_map[list(self.v_end_allele_correction_map)[0]])

    # Sequence Simulation
    def simulate_sequence(self):
        # Sample Sequence
        gen = LightChainSequence.create_random(self.dataconfig)

        gen.mutate(self.mutation_model)

        data = {
            "sequence": gen.mutated_seq,
            "v_sequence_start": gen.v_seq_start,
            "v_sequence_end": gen.v_seq_end,
            "j_sequence_start": gen.j_seq_start,
            "j_sequence_end": gen.j_seq_end,
            "v_allele": [gen.v_allele.name],
            "j_allele": [gen.j_allele.name],
            'mutation_rate': gen.mutation_freq,
            'v_trim_5': gen.v_trim_5,
            'v_trim_3': gen.v_trim_3,
            'j_trim_5': gen.j_trim_5,
            'j_trim_3': gen.j_trim_3,
            'type': self.chain_type,
            'corruption_event': 'no-corruption',
            'corruption_add_amount': 0,
            'corruption_remove_amount': 0,
            'mutations': {pos: gen.mutations[pos] for pos in sorted(gen.mutations)},  # sort the mutations by position
            "Ns": dict()
        }
        return data

    def fix_v_position_after_trimming_index_ambiguity(self, simulation):
        # Extract Current V Metadata
        v_start, v_end = simulation['v_sequence_start'], simulation['v_sequence_end']
        v_allele_remainder = simulation['sequence'][v_start:v_end]
        v_allele_ref = self.v_dict[simulation['v_allele'][0]]

        # Get the junction inserted after trimming to the sequence
        junction_3 = simulation['sequence'][v_end:simulation['j_sequence_start']]

        # Get the trimming lengths
        v_trim_3 = simulation['v_trim_3']

        # Get the trimmed off sections from the reference
        trimmed_3 = v_allele_ref[len(v_allele_ref) - v_trim_3:]

        # check for overlap between generated junction and reference in the 3' trim
        for a, b in zip(trimmed_3, junction_3):
            # in case the current poistion in the junction matches the reference exapnd the d segment
            if a == b:
                v_end += 1
            else:  # if the continuous streak is broken or non-existent break!
                break

        simulation['v_sequence_end'] = v_end

    def fix_j_position_after_trimming_index_ambiguity(self, simulation):
        # Extract Current J Metadata
        j_start, j_end = simulation['j_sequence_start'], simulation['j_sequence_end']
        j_allele_remainder = simulation['sequence'][j_start:j_end]
        j_allele_ref = self.j_dict[simulation['j_allele'][0]]

        # Get the junction inserted after trimming to the sequence
        junction_5 = simulation['sequence'][simulation['v_sequence_end']:j_start]

        # Get the trimming lengths
        j_trim_5 = simulation['j_trim_5']

        # Get the trimmed off sections from the reference
        trimmed_5 = j_allele_ref[:j_trim_5]

        # check for overlap between generated junction and reference in the 5' trim
        for a, b in zip(trimmed_5[::-1], junction_5[::-1]):
            # in case the current poistion in the junction matches the reference exapnd the d segment
            if a == b:
                j_start -= 1
            else:  # if the continuous streak is broken or non-existent break!
                break

        simulation['j_sequence_start'] = j_start

    def remove_event(self, simulated):
        # Update Simulation Metadata
        simulated['corruption_event'] = 'remove'
        # Sample how much to remove
        v_length = simulated['v_sequence_end'] - simulated['v_sequence_start']
        amount_to_remove = self._sample_nucleotide_remove_distribution(v_length)
        # remove from the start of the sequence the sampled amount
        simulated['sequence'] = simulated['sequence'][amount_to_remove:]
        # Update Simulation Metadata
        simulated['corruption_remove_amount'] = amount_to_remove
        # Update mutation log, remove the mutations that were removed with the remove event
        # and while updating the mutations log correct the position of the mutations accordingly
        simulated['mutations'] = {i - amount_to_remove: j for i, j in simulated['mutations'].items() if
                                  i > amount_to_remove}
        # Adjust mutation rate
        self.correct_mutation_rate(simulated)

        # Adjust Start/End Position Accordingly
        simulated['v_sequence_start'] = 0
        simulated['v_sequence_end'] -= amount_to_remove
        simulated['j_sequence_start'] -= amount_to_remove
        simulated['j_sequence_end'] -= amount_to_remove

        # Correction - Add All V Alleles That Cant be Distinguished Based on the Amount Cut from the V Allele
        self.correct_for_v_start_cut(simulated)

    def add_event(self, simulated, amount=None):
        # Update Simulation Metadata
        simulated['corruption_event'] = 'add'
        # Sample the Amount to Add by default, if a spesific value is given via the amount variable,use it.
        amount_to_add = self._sample_nucleotide_add_distribution() if amount is None else amount
        # Sample the method by which addition will be made
        method = self._sample_corruption_add_method()
        # Modify the sequence
        modified_sequence = method(amount_to_add, simulated['sequence'])
        # Validate the modified sequence, make sure we didn't over add pass our max sequence size
        modified_sequence, amount_to_add = self.validate_sequence_length_after_addition(modified_sequence,
                                                                                        amount_to_add)
        # Update Simulation Sequence
        simulated['sequence'] = modified_sequence

        # Update Simulation Metadata
        simulated['corruption_add_amount'] = amount_to_add
        # Update mutation log positions
        simulated['mutations'] = {i + amount_to_add: j for i, j in simulated['mutations'].items()}

        # Adjust Start/End Position Accordingly
        simulated['v_sequence_start'] += amount_to_add
        simulated['v_sequence_end'] += amount_to_add
        simulated['j_sequence_start'] += amount_to_add
        simulated['j_sequence_end'] += amount_to_add

    def remove_before_add_event(self, simulated):
        # ----- REMOVE PART -----#
        # Sample how much to remove
        self.remove_event(simulated)

        # Update Simulation Metadata
        simulated['corruption_event'] = 'remove_before_add'

        # ----- ADD PART -----#
        # Sample how much to add after removal occurred
        amount_to_add = self._sample_nucleotide_add_after_remove_distribution()
        self.add_event(simulated, amount=amount_to_add)

    def simulate_augmented_sequence(self):
        # 1. Simulate a Sequence from AIRRship
        simulated = self.simulate_sequence()

        # 1.1 Correction - Correct Start/End Positions Based on Generated Junctions
        self.fix_v_position_after_trimming_index_ambiguity(simulated)
        self.fix_j_position_after_trimming_index_ambiguity(simulated)

        # 1.2 Correction - Add All V Alleles That Cant be Distinguished Based on the Amount Cut from the V Allele
        self.correct_for_v_end_cut(simulated)

        # 2. Corrupt Begging of Sequence ( V Start )
        if self.perform_corruption():
            # Inside this method, based on the corruption event we will also adjust the respective ground truth v
            # alleles
            self.corrupt_sequence_beginning(simulated)

        # 3. Add N's
        self.insert_Ns(simulated)

        self.process_before_return(simulated)

        return simulated


class LightChainKappaLambdaSequenceAugmentor:
    def __init__(self, lambda_dataconfig: DataConfig, kappa_dataconfig: DataConfig,
                 lambda_args: SequenceAugmentorArguments = SequenceAugmentorArguments(),
                 kappa_args: SequenceAugmentorArguments = SequenceAugmentorArguments()):
        self.lambda_augmentor = LightChainSequenceAugmentor(lambda_dataconfig, lambda_args)
        self.kappa_augmentor = LightChainSequenceAugmentor(kappa_dataconfig, kappa_args)
        self.kappa_lambda_ratio = lambda_args.kappa_lambda_ratio

    def simulate_augmented_sequence(self):
        chain_type = np.random.choice([LightChainType.KAPPA, LightChainType.LAMBDA], size=1, p=[self.kappa_lambda_ratio,
                                                                                                1 - self.kappa_lambda_ratio]).item()
        if chain_type == LightChainType.KAPPA:
            return self.kappa_augmentor.simulate_augmented_sequence()
        else:
            return self.lambda_augmentor.simulate_augmented_sequence()
