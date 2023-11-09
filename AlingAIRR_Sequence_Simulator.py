import enum
import pickle
import random
from dataclasses import dataclass
import numpy as np
import scipy.stats as st
from airrship.create_repertoire import generate_sequence, load_data, global_genotype


# Corruption Events
class Event(enum.Enum):
    Remove = 1
    Add = 2
    Remove_Before_Add = 3


# Arg Class
@dataclass
class SequenceSimulatorArguments:
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
    """
    min_mutation_rate: float = 0.003
    max_mutation_rate: float = 0.25
    n_ratio: float = 0.02
    v_allele_map_path: str = '/your/path/'
    max_sequence_length: int = 512
    mutation_model: str = "s5f"
    nucleotide_add_coefficient: float = 210
    nucleotide_remove_coefficient: float = 310
    nucleotide_add_after_remove_coefficient: float = 50
    random_sequence_add_proba: float = 1
    single_base_stream_proba: float = 0
    duplicate_leading_proba: float = 0
    random_allele_proba: float = 0
    corrupt_proba: float = 0.7
    short_d_length: int = 3


class SequenceSimulator:

    def __init__(self, args: SequenceSimulatorArguments = SequenceSimulatorArguments()):

        # AIRRship Parameters
        self.min_mutation_rate = args.min_mutation_rate
        self.max_mutation_rate = args.max_mutation_rate
        self.data_dict = load_data()
        self.locus = global_genotype()
        self.mutation_model = args.mutation_model

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
        self.v_allele_map_path = args.v_allele_map_path
        self.v_alleles = self.locus[0]["V"]
        self.max_v_length = max(map(lambda x: len(x.ungapped_seq), self.v_alleles))
        self.max_sequence_length = args.max_sequence_length
        self.short_d_length = args.short_d_length

        # Loading Routines
        self.load_correction_maps()

    # Loading Routines
    def load_correction_maps(self):
        """
        This will load the V start and V end maps that tell us given the amount removed from the start or end
        of a given allele what optionion are equally likely
        :return:
        """
        with open(self.v_allele_map_path + 'V_ALLELE_SIMILARITY_MAP.pkl', 'rb') as h:
            self.v_start_allele_correction_map = pickle.load(h)
            self.max_v_start_correction_map_value = max(
                self.v_start_allele_correction_map[list(self.v_start_allele_correction_map)[0]])

        with open(self.v_allele_map_path + 'V_ALLELE_SIMILARITY_MAP_AT_END.pkl', 'rb') as h:
            self.v_end_allele_correction_map = pickle.load(h)
            self.max_v_end_correction_map_value = max(
                self.v_end_allele_correction_map[list(self.v_end_allele_correction_map)[0]])

    # Noise Introducing Methods
    def insert_Ns(self, string):
        num_replacements = int(len(string) * self.n_ratio)
        nucleotides_list = list(string)

        for _ in range(num_replacements):
            index = random.randint(0, len(nucleotides_list) - 1)
            nucleotides_list[index] = "N"

        return ''.join(nucleotides_list)

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
        # Adjust Start/End Position Accordingly
        simulated['v_sequence_start'] = 0
        simulated['v_sequence_end'] -= amount_to_remove
        simulated['d_sequence_start'] -= amount_to_remove
        simulated['d_sequence_end'] -= amount_to_remove
        simulated['j_sequence_start'] -= amount_to_remove
        simulated['j_sequence_start'] -= amount_to_remove

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

        # Adjust Start/End Position Accordingly
        simulated['v_sequence_start'] = amount_to_add
        simulated['v_sequence_end'] += amount_to_add
        simulated['d_sequence_start'] += amount_to_add
        simulated['d_sequence_end'] += amount_to_add
        simulated['j_sequence_start'] += amount_to_add
        simulated['j_sequence_start'] += amount_to_add

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

    def correct_for_v_start_cut(self, simulated):
        removed = simulated['corruption_remove_amount']
        equivalent_alleles = self.v_start_allele_correction_map[simulated['v_allele'][0]][
            min(removed, self.max_v_start_correction_map_value)]
        simulated['v_allele'] = simulated['v_allele'] + equivalent_alleles

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

    # AIRRship
    def query_airrship(self):
        mutation_rate = np.random.uniform(self.min_mutation_rate, self.max_mutation_rate, 1).item()

        gen = generate_sequence(self.locus, self.data_dict, mutate=True,
                                mutation_rate=mutation_rate,
                                shm_flat=self.mutation_model.lower() != 's5f', flat_usage='allele')
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
            'mutation_rate': mutation_rate,
            'corruption_event': 'no-corruption',
            'corruption_add_amount': 0,
            'corruption_remove_amount': 0,
        }
        return data

    # Interface
    def get_sequence(self):
        # 1. Simulate a Sequence from AIRRship
        simulated = self.query_airrship()

        # 1.1 Correction - Add All V Alleles That Cant be Distinguished Based on the Amount Cut from the V Allele
        self.correct_for_v_end_cut(simulated)

        # 2. Add N's
        simulated['sequence'] = self.insert_Ns(simulated['sequence'])
        # 2.1 Adjust mutation rate to account for added "N's" by adding the ratio of N's added to the mutation rate
        simulated['mutation_rate'] = simulated['mutation_rate'] + self.n_ratio
        # 2.2 Consider Adding Correction For the True Allele Set Based on the Positions the N's were inserted too

        # 3. Corrupt Begging of Sequence ( V Start )
        if self.perform_corruption():
            # Inside this method, based on the corruption event we will also adjust the respective ground truth v
            # alleles
            self.corrupt_sequence_beginning(simulated)

        # 4. Adjust D Allele , if the simulated length is smaller than short_d_length
        # class property, change the d allele to the "Short-D" label
        self.short_d_validation(simulated)

        # To-Do adjust mutation rate based on how many actually left after start removal and + N insertion

        # Convert Allele From List to Comma Seperated String
        for gene in ['v','d','j']:
            simulated[f'{gene}_allele'] = ','.join(simulated[f'{gene}_allele'])

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
