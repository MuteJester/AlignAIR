from abc import ABC, abstractmethod
from SequenceSimulation.utilities.data_config import DataConfig
from dataclasses import dataclass
import enum
from SequenceSimulation.mutation import MutationModel, S5F
import scipy.stats as st
import pickle
import random
import base64
import numpy as np


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
    custom_mutation_model_path: str = None
    nucleotide_add_coefficient: float = 210
    nucleotide_remove_coefficient: float = 310
    nucleotide_add_after_remove_coefficient: float = 50
    random_sequence_add_proba: float = 1
    single_base_stream_proba: float = 0
    duplicate_leading_proba: float = 0
    random_allele_proba: float = 0
    corrupt_proba: float = 0.7
    short_d_length: int = 3
    kappa_lambda_ratio: float = 0.5
    save_mutations_record: bool = False
    save_ns_record: bool = False


class SequenceAugmentorBase(ABC):
    alleles_used = None

    def __init__(self, dataconfig: DataConfig, args: SequenceAugmentorArguments = SequenceAugmentorArguments()):

        # Sequence Generation Parameters
        self.dataconfig = dataconfig
        self.min_mutation_rate = args.min_mutation_rate
        self.max_mutation_rate = args.max_mutation_rate
        if args.custom_mutation_model_path is not None:
            self.mutation_model = args.mutation_model(self.min_mutation_rate, self.max_mutation_rate,
                                                      custom_model=args.custom_mutation_model_path)
        else:
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
        self.chain_type = None
        self.save_mutations_record = args.save_mutations_record
        self.save_ns_record = args.save_ns_record
        self.v_alleles = sorted([i for j in self.dataconfig.v_alleles for i in self.dataconfig.v_alleles[j]],
                                key=lambda x: x.name)

        self.j_alleles = sorted([i for j in self.dataconfig.j_alleles for i in self.dataconfig.j_alleles[j]],
                                key=lambda x: x.name)
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.v_alleles}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.j_alleles}
        self.max_v_length = max(map(lambda x: len(x.ungapped_seq), self.v_alleles))
        self.max_sequence_length = args.max_sequence_length

        # Loading Routines
        self.load_correction_maps()

    # Loading Routines
    @abstractmethod
    def load_correction_maps(self):
        pass

    def get_original_index(self, corrupted_index, simulated):
        """
        Translates an index from the corrupted sequence back to the corresponding index in the original sequence.
        """
        corruption_event = simulated['corruption_event']
        amount_added = simulated['corruption_add_amount']
        amount_removed = simulated['corruption_remove_amount']

        if corruption_event == 'add':
            # If characters were added at the beginning, the original index is shifted to the right
            return corrupted_index - amount_added if corrupted_index >= amount_added else None
        elif corruption_event == 'remove':
            # If characters were removed from the beginning, the original index is shifted to the left
            return corrupted_index + amount_removed
        elif corruption_event == 'remove_before_add':
            # Combined effect of removal and addition
            adjusted_index = corrupted_index + amount_removed
            return adjusted_index - amount_added if adjusted_index >= amount_added else None
        else:
            # If no corruption or unknown corruption event, assume the index is unchanged
            return corrupted_index

    def get_allele_spesific_n_positions(self, simulated, n_positions, allele):
        return [i for i in n_positions if
                simulated[f'{allele}_sequence_start'] <= i <= simulated[f'{allele}_sequence_end']]

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

        # Get All the N Poistion Inserted to the V Allele
        v_allele_n_positions = self.get_allele_spesific_n_positions(simulated, list(simulated['Ns']), 'v')
        # Get Original Positions For Correction
        v_allele_n_positions_original = [self.get_original_index(i, simulated) for i in v_allele_n_positions]

        indistinguishable_v_alleles = self.v_n_ambiguity_comparer.find_indistinguishable_alleles(
            simulated['v_allele'][0], v_allele_n_positions_original)
        simulated['v_allele'] = simulated['v_allele'] + list(
            set(indistinguishable_v_alleles) - set(simulated['v_allele']))

        # Adjust mutation rate to account for added "N's" by adding the ratio of N's added to the mutation rate
        # because we randomly sample position we might get a ratio of N's less than what was defined by the n_ratio
        # property, thus we recalculate the actual inserted ration of N's
        # Also make sure we only count the N's the were inserted in the sequence and not in the added slack
        # as we might consider all the slack as noise in general
        Ns_in_pure_sequence = [i for i in simulated['Ns'] if
                               simulated['v_sequence_start'] <= i <= simulated['j_sequence_end']]
        pure_sequence_length = simulated['j_sequence_end'] - simulated['v_sequence_start']

        # exception handling
        if pure_sequence_length == 0:
            # handle division by zero
            simulated_n_ratio = 0
        else:
            simulated_n_ratio = len(Ns_in_pure_sequence) / pure_sequence_length
        simulated['mutation_rate'] += simulated_n_ratio

    def remove_event(self, simulated, autocorrect=True):
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
        if autocorrect:
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
        self.remove_event(simulated, autocorrect=False)  # Dont Correct For Removed Section YET! First Lets Add

        # ----- ADD PART -----#
        # Sample how much to add after removal occurred
        amount_to_add = self._sample_nucleotide_add_after_remove_distribution()
        self.add_event(simulated, amount=amount_to_add)

        # Update Simulation Metadata
        simulated['corruption_event'] = 'remove_before_add'

        # Check If The Addition Recreated Some of the Removed Section by Chance
        self.correct_for_v_start_add(simulated)
        # Correct for the removed section of the V now that we have check for the reconstruction by chance
        self.correct_for_v_start_cut(simulated)

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
        simulated['v_allele'] = simulated['v_allele'] + list(set(equivalent_alleles) - set(simulated['v_allele']))

    def correct_for_v_start_cut(self, simulated):
        removed = simulated['corruption_remove_amount']
        equivalent_alleles = self.v_start_allele_correction_map[simulated['v_allele'][0]][
            min(removed, self.max_v_start_correction_map_value)]
        simulated['v_allele'] = simulated['v_allele'] + list(set(equivalent_alleles) - set(simulated['v_allele']))

    def correct_for_v_start_add(self, simulated):
        """
        this method will take the simulated sequence and check whether it had an Remove event and then an Add event,
        In such a case we should test the add section and validated that it didnt recreate the section that was removed
        if it does, we should adjust the v start and position accordingly.
        :param simulated:
        :return:
        """

        if simulated['corruption_event'] == 'remove_before_add':
            amount_added = simulated['corruption_add_amount']
            amount_removed = simulated['corruption_remove_amount']

            add_section = simulated['sequence'][:amount_added]
            removed_section = self.v_dict[simulated['v_allele'][0]][:amount_removed]

            min_length = min(len(add_section), len(removed_section))
            to_adjust = 0
            for i in range(1, min_length + 1):
                # Compare characters from the end
                if add_section[-i] == removed_section[-i]:
                    to_adjust += 1
                else:  # Mismatch found, halt the iteration
                    break

            # Adjust V Start
            simulated['v_sequence_start'] -= to_adjust
            # Adjust Removed Amount
            simulated['corruption_remove_amount'] -= to_adjust

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
    @abstractmethod
    def simulate_sequence(self):
        """
        return format
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
        :return:
        """
        pass

    # Interface
    def process_before_return(self, simulated):
        """
        this method makes final adjustments to the output format
        :return:
        """
        # Convert Allele From List to Comma Seperated String
        for gene in self.alleles_used:
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

    @abstractmethod
    def simulate_augmented_sequence(self):
        pass

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
