import base64
import enum
import pickle
import random
from dataclasses import dataclass
import numpy as np
import scipy.stats as st

from SequenceSimulation.mutation import MutationModel, S5F
from SequenceSimulation.sequence import HeavyChainSequence
from SequenceSimulation.utilities.data_config import DataConfig
from airrship.create_repertoire import generate_sequence, load_data, global_genotype
import base64


# Corruption Events
class Event(enum.Enum):
    Remove = 1
    Add = 2
    Remove_Before_Add = 3


# Arg Class
@dataclass
class SequenceAugmentorArguments:
    """
    Arguments for the SequenceSimulator class.

    min_mutation_rate: The minimum mutation rate.
    max_mutation_rate: The maximum mutation rate.
    n_ratio: The ratio of N's inserted to the sequence.
    v_allele_map_path: The path to the V allele map files.
    max_sequence_length: The maximum sequence length.
    mutation_model: The mutation model to use.
    nucleotide_add_coefficient: The coefficient for the nucleotide add distribution.
    nucleotide_remove_coefficient: The coefficient for the nucleotide remove distribution.
    nucleotide_add_after_remove_coefficient: The coefficient for the nucleotide add after remove distribution.
    random_sequence_add_proba: The probability of adding a random sequence.
    single_base_stream_proba: The probability of adding a single base stream.
    duplicate_leading_proba: The probability of duplicating the leading base.
    random_allele_proba: The probability of adding a random allele.
    corrupt_proba: The probability of corrupting the sequence from the start.
    short_d_length: The minimum length required from the D allele to not be tagged as "Short-D"
    save_mutations_record: Whether to save the mutations that were in the sequence or not (saved as base64 dictionary)
    save_ns_record: Whether to save the N's that were in the sequence or not (saved as base64 dictionary)
    """
    min_mutation_rate: float = 0.003
    max_mutation_rate: float = 0.25
    n_ratio: float = 0.02
    max_sequence_length: int = 512
    mutation_model: MutationModel = S5F
    nucleotide_add_coefficient: float = 210
    nucleotide_remove_coefficient: float = 310
    nucleotide_add_after_remove_coefficient: float = 50
    random_sequence_add_proba: float = 1
    single_base_stream_proba: float = 0
    duplicate_leading_proba: float = 0
    random_allele_proba: float = 0
    corrupt_proba: float = 0.7
    short_d_length: int = 3
    save_mutations_record: bool = False
    save_ns_record: bool = False


class SequenceAugmentor:

    def __init__(self, dataconfig: DataConfig, args: SequenceAugmentorArguments = SequenceAugmentorArguments()):

        # Sequence Generation Parameters
        self.dataconfig = dataconfig
        self.min_mutation_rate = args.min_mutation_rate
        self.max_mutation_rate = args.max_mutation_rate
        self.mutation_model = args.mutation_model(self.min_mutation_rate, self.max_mutation_rate)

        # Noising Parameters
        self.n_ratio = args.n_ratio
        self.corrupt_proba = args.corrupt_proba
        self.nucleotide_add_distribution = st.beta(2, 3)
        self.nucleotide_remove_distribution = st.beta(2, 3)
        self.nucleotide_add_after_remove_distribution = st.beta(1, 3)

        self.nucleotide_add_coef = args.nucleotide_add_coefficient
        self.nucleotide_remove_coef = args.nucleotide_remove_coefficient
        self.nucleotide_add_after_remove_coef = args.nucleotide_add_after_remove_coefficient

        self.random_sequence_add_proba = args.random_sequence_add_proba
        self.single_base_stream_proba = args.single_base_stream_proba
        self.duplicate_leading_proba = args.duplicate_leading_proba
        self.random_allele_proba = args.random_allele_proba

        # Class Misc
        self.save_mutations_record = args.save_mutations_record
        self.save_ns_record = args.save_ns_record
        self.v_alleles = sorted([i for j in self.dataconfig.v_alleles for i in self.dataconfig.v_alleles[j]],
                                key=lambda x: x.name)
        self.d_alleles = sorted([i for j in self.dataconfig.d_alleles for i in self.dataconfig.d_alleles[j]],
                                key=lambda x: x.name)
        self.j_alleles = sorted([i for j in self.dataconfig.j_alleles for i in self.dataconfig.j_alleles[j]],
                                key=lambda x: x.name)
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.v_alleles}
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.d_alleles}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.j_alleles}
        self.max_v_length = max(map(lambda x: len(x.ungapped_seq), self.v_alleles))
        self.max_sequence_length = args.max_sequence_length
        self.short_d_length = args.short_d_length

        # Loading Routines
        self.load_correction_maps()

    # Loading Routines
    def load_correction_maps(self):
        from importlib import resources
        """
        This will load the V start and V end maps that tell us given the amount removed from the start or end
        of a given allele what optionion are equally likely
        :return:
        """
        with resources.path('SequenceSimulation.data', 'V_ALLELE_SIMILARITY_MAP.pkl') as data_path:
            with open(data_path, 'rb') as h:
                self.v_start_allele_correction_map = pickle.load(h)
                self.max_v_start_correction_map_value = max(
                    self.v_start_allele_correction_map[list(self.v_start_allele_correction_map)[0]])

        with resources.path('SequenceSimulation.data', 'V_ALLELE_SIMILARITY_MAP_AT_END.pkl') as data_path:
            with open(data_path, 'rb') as h:
                self.v_end_allele_correction_map = pickle.load(h)
                self.max_v_end_correction_map_value = max(
                    self.v_end_allele_correction_map[list(self.v_end_allele_correction_map)[0]])

        with resources.path('SequenceSimulation.data', 'd_allele_trim_map.pkl') as data_path:
            with open(data_path, 'rb') as h:
                self.d_trim_correction_map = pickle.load(h)

    # Noise Introducing Methods
    def insert_Ns(self, simulated):
        sequence = simulated['sequence']
        # Calculate how many Ns we should insert
        num_replacements = int(len(sequence) * self.n_ratio)
        nucleotides_list = list(sequence)

        for _ in range(num_replacements):
            # Get random position in the sequence
            index = random.randint(0, len(nucleotides_list) - 1)
            # Log the N insertion event in the sequence metadata
            simulated['Ns'][index] = nucleotides_list[index] + '>' + 'N'
            # Make the N insertion
            nucleotides_list[index] = "N"

        # Concatenate the list back into a string
        simulated['sequence'] = ''.join(nucleotides_list)

        # Sort the N's insertion log
        simulated['Ns'] = {pos: simulated['Ns'][pos] for pos in sorted(simulated['Ns'])}

        # Adjust mutation rate to account for added "N's" by adding the ratio of N's added to the mutation rate
        # because we randomly sample position we might get a ratio of N's less than what was defined by the n_ratio
        # property, thus we recalculate the actual inserted ration of N's
        # Also make sure we only count the N's the were inserted in the sequence and not in the added slack
        # as we might consider all the slack as noise in general
        Ns_in_pure_sequence = [i for i in simulated['Ns'] if
                               simulated['v_sequence_start'] <= i <= simulated['j_sequence_end']]
        pure_sequence_length = simulated['j_sequence_end'] - simulated['v_sequence_start']
        simulated_n_ratio = len(Ns_in_pure_sequence) / pure_sequence_length
        simulated['mutation_rate'] += simulated_n_ratio

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
        simulated['d_sequence_start'] -= amount_to_remove
        simulated['d_sequence_end'] -= amount_to_remove
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
        simulated['d_sequence_start'] += amount_to_add
        simulated['d_sequence_end'] += amount_to_add
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

    def corrupt_sequence_beginning(self, simulated):
        # Sample Corruption Event
        event = self.sample_random_event()
        if event is Event.Remove:
            self.remove_event(simulated)
        elif event is Event.Add:
            self.add_event(simulated)

        elif event is Event.Remove_Before_Add:
            self.remove_before_add_event(simulated)
        else:
            raise ValueError(f'Unknown Corruption Event {event}')

    # Different V Start "Add" Event Scenarios
    @staticmethod
    def random_nucleotides(amount, sequence):
        random_seq = ''.join(random.choices(['A', 'T', 'C', 'G'], k=amount))
        return random_seq + sequence

    @staticmethod
    def duplicate_leading(amount, sequence):
        cap = amount if amount < len(sequence) else len(sequence) - 1
        return sequence[:cap] + sequence

    def random_allele_section(self, amount, sequence):
        random_allele = random.choice(self.v_alleles).ungapped_seq.upper()
        cap = amount if amount < len(random_allele) else len(random_allele) - 1
        return random_allele[:cap] + sequence

    @staticmethod
    def single_base_stream(amount, sequence):
        random_base = random.choice(['A', 'T', 'G', 'C', 'N']) * amount
        return random_base + sequence

    # Correction Functions
    def correct_for_v_end_cut(self, simulated):
        equivalent_alleles = self.v_end_allele_correction_map[simulated['v_allele'][0]][
            min(simulated['v_sequence_end'], self.max_v_end_correction_map_value)]
        simulated['v_allele'] = simulated['v_allele'] + equivalent_alleles

    def correct_for_d_trims(self, simulated):
        # Get the 5' and 3' trims of the d allele in the simulated sequence
        trim_5 = simulated['d_trim_5']
        trim_3 = simulated['d_trim_3']
        # infer the precalculated map what alleles should be the ground truth for this sequence based on the trim
        simulated['d_allele'] = list(self.d_trim_correction_map[simulated['d_allele'][0]][(trim_5, trim_3)])

    def correct_for_v_start_cut(self, simulated):
        removed = simulated['corruption_remove_amount']
        equivalent_alleles = self.v_start_allele_correction_map[simulated['v_allele'][0]][
            min(removed, self.max_v_start_correction_map_value)]
        simulated['v_allele'] = simulated['v_allele'] + equivalent_alleles

    def correct_mutation_rate(self, simulated):
        """
        After we insert N's into the sequence or make any kind of removal or modification to the sequence, we might
        change the actual mutation rate by overwriting / changing a simulated mutation, thus the porpuse of this function
        is the update the mutation log appropriately as well as the mutation rate in the simulation log so that it will
        match the actual ratio of mutation in the simulataed sequence
        :param simulated:
        :return:
        """

        # recalculate the mutation ratio based on the current mutation log
        n_mutations = len(simulated['mutations'])
        # the actual sequence length without any noise at the start or at the end
        sequence_length = simulated['j_sequence_end'] - simulated['v_sequence_start']
        mutation_ratio = n_mutations / sequence_length
        # update the mutation_rate in the simulation log
        simulated['mutation_rate'] = mutation_ratio

    def validate_sequence_length_after_addition(self, sequence, added):
        # In case we added too many nucleotides to the beginning while corrupting the sequence, remove the slack
        if len(sequence) > self.max_sequence_length:
            # Calculate how much did we over add to based on our maximum sequence length
            slack = len(sequence) - self.max_sequence_length
            # remove the slack from the begging and update the added variable to match the final added amount
            return sequence[slack:], added - slack
        else:
            return sequence, added

    def short_d_validation(self, simulated):
        d_length = simulated['d_sequence_end'] - simulated['d_sequence_start']
        if d_length < self.short_d_length:
            simulated['d_allele'] = ['Short-D']

    def fix_v_position_after_trimming_index_ambiguity(self, simulation):
        # Extract Current V Metadata
        v_start, v_end = simulation['v_sequence_start'], simulation['v_sequence_end']
        v_allele_remainder = simulation['sequence'][v_start:v_end]
        v_allele_ref = self.v_dict[simulation['v_allele'][0]]

        # Get the junction inserted after trimming to the sequence
        junction_3 = simulation['sequence'][v_end:simulation['d_sequence_start']]

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

    def fix_j_position_after_trimming_index_ambiguity(self, simulation):
        # Extract Current J Metadata
        j_start, j_end = simulation['j_sequence_start'], simulation['j_sequence_end']
        j_allele_remainder = simulation['sequence'][j_start:j_end]
        j_allele_ref = self.j_dict[simulation['j_allele'][0]]

        # Get the junction inserted after trimming to the sequence
        junction_5 = simulation['sequence'][simulation['d_sequence_end']:j_start]

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

    # Sequence Simulation
    def simulate_sequence(self):

        if self.dataconfig.d_alleles is None:  # If No D, Then it most be light chain
            pass
        else:  # it is a heavy chain sequence
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

    # Interface
    def process_before_return(self, simulated):
        """
        this method makes final adjustments to the output format
        :return:
        """
        # Convert Allele From List to Comma Seperated String
        for gene in ['v', 'd', 'j']:
            simulated[f'{gene}_allele'] = ','.join(simulated[f'{gene}_allele'])

        # convert mutations and N log to base64 for proper tabular preservation
        if self.save_ns_record:
            simulated['Ns'] = base64.b64encode(str(simulated['Ns']).encode('ascii'))
        else:
            simulated.pop('Ns')

        if self.save_mutations_record:
            simulated['mutations'] = base64.b64encode(str(simulated['mutations']).encode('ascii'))
        else:
            simulated.pop('mutations')

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

    # Misc Methods
    def _sample_nucleotide_add_distribution(self):
        sample = (self.nucleotide_add_coef * self.nucleotide_add_distribution.rvs(size=1)).astype(int)
        return sample.item()

    def _sample_nucleotide_remove_distribution(self, v_length):
        # Sample amount based on predefined distribution
        sample = (self.nucleotide_remove_coef * self.nucleotide_remove_distribution.rvs(size=1)).astype(int).item()
        # make sure that no matter how much we get from sampling the predefined distribution we wont get a value
        # larger than the total v length we have in our sequence
        sample = min(sample, v_length)
        return sample

    def _sample_nucleotide_add_after_remove_distribution(self):
        sample = (self.nucleotide_add_after_remove_coef * self.nucleotide_add_after_remove_distribution.rvs(
            size=1)).astype(int)
        return sample.item()

    def perform_corruption(self):
        return bool(np.random.binomial(1, self.corrupt_proba))

    @staticmethod
    def sample_random_event():
        return np.random.choice([Event.Remove, Event.Add, Event.Remove_Before_Add], size=1, p=[0.4, 0.4, 0.2]).item()

    def _sample_corruption_add_method(self):
        method = random.choices([self.random_nucleotides, self.single_base_stream, self.duplicate_leading,
                                 self.random_allele_section],
                                weights=[self.random_sequence_add_proba, self.single_base_stream_proba,
                                         self.duplicate_leading_proba, self.random_allele_proba], k=1)[0]
        return method

    @property
    def columns(self):
        return list(self.simulate_augmented_sequence())
