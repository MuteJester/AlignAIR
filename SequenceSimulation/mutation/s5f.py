import random
from SequenceSimulation.mutation.mutation_model import MutationModel
import pickle


class Nucleotide:
    def __init__(self, value):
        self.value = value
        self.adjacent = []  # References to adjacent nucleotides

    def update_value(self, new_value, update_callback=None):
        self.value = new_value
        if update_callback:
            for five_mer in self.adjacent:
                update_callback(five_mer)

    def __repr__(self):
        return self.value


class FiveMER:
    def __init__(self, nucleotides):
        self.nucleotides = nucleotides  # List of Nucleotide objects
        for nuc in self.nucleotides:
            nuc.adjacent.append(self)
        self.sequence = ''.join([nuc.value for nuc in nucleotides])
        self.position = None
        self.likelihood = 0
        self.modified = False

    def update_sequence(self, mutability=None):
        self.sequence = ''.join([nuc.value for nuc in self.nucleotides])
        if mutability is not None:
            self.likelihood = self.likelihood if self.sequence not in mutability else mutability[self.sequence]

    def change_center(self, new_value, mutability=None):
        center_nucleotide = self.nucleotides[2]  # Assuming 0-indexed, 2 is the center
        # Pass the update callback to update_value
        center_nucleotide.update_value(new_value, lambda fm: fm.update_sequence(mutability))
        self.modified = True

    def __repr__(self):
        return ''.join([i.value for i in self.nucleotides])

    def __eq__(self, other):
        if isinstance(other, FiveMER):
            return self.sequence == other.sequence
        elif isinstance(other, str):
            return self.sequence == other
        else:
            raise TypeError("Unsupported comparison between FiveMER and {}".format(type(other)))

    @staticmethod
    def create_five_mers(dna_sequence, mutability=None):
        # Step 1: Pad the sequence
        padded_sequence = 'NN' + dna_sequence + 'NN'

        # Step 2: Create Nucleotide objects for the padded sequence
        nucleotides = [Nucleotide(nuc) for nuc in padded_sequence]
        # Step 3: Group nucleotides into FiveMER objects
        five_mers = []
        for i in range(len(dna_sequence)):  # Iterate based on the original sequence length
            five_mer_nucleotides = nucleotides[i:i + 5]
            five_mer = FiveMER(five_mer_nucleotides)
            five_mer.position = i  # Position of the original second nucleotide (central in unpadded)

            # if mutability map was supplied
            if mutability is not None:
                str_five_mer = five_mer.sequence
                five_mer.likelihood = 0 if str_five_mer not in mutability else mutability[str_five_mer]

            five_mers.append(five_mer)

        return five_mers

    @staticmethod
    def five_mers_to_dna(five_mers):
        if not five_mers:
            return ""

        # Start with the central nucleotide of the first FiveMER (skipping initial padding)
        dna_sequence = five_mers[0].nucleotides[2].value

        # Iterate over the remaining FiveMERs and append only the last nucleotide of each
        for five_mer in five_mers[1:-1]:  # Skip the last FiveMER with padding
            dna_sequence += five_mer.nucleotides[2].value

        # Add the central nucleotide of the last FiveMER (which is the actual last nucleotide of the DNA)
        dna_sequence += five_mers[-1].nucleotides[2].value

        return dna_sequence


class S5F(MutationModel):
    def __init__(self, min_mutation_rate=0, max_mutation_rate=0,custom_model=None):
        self.targeting = None
        self.substitution = None
        self.mutability = None
        self.max_mutation_rate = max_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.bases = {'A', 'T', 'C', 'G'}
        self.loaded_metadata = False
        self.custom_model = custom_model

    def load_metadata(self, sequence):
        from SequenceSimulation.sequence import HeavyChainSequence
        from SequenceSimulation.sequence.light_chain import LightChainSequence
        from importlib import resources

        if self.custom_model is None:
            if type(sequence) == HeavyChainSequence:
                with resources.path('SequenceSimulation.data', 'HH_S5F_META.pkl') as data_path:
                    with open(data_path, 'rb') as h:
                        self.mutability, self.substitution, self.targeting = pickle.load(h)
            elif type(sequence) == LightChainSequence:
                with resources.path('SequenceSimulation.data', 'HKL_S5F_META.pkl') as data_path:
                    with open(data_path, 'rb') as h:
                        self.mutability, self.substitution, self.targeting = pickle.load(h)
            else:
                raise ValueError('Unsupported Sequence Type')
        else:
            with open(self.custom_model, 'rb') as h:
                self.mutability, self.substitution, self.targeting = pickle.load(h)

    def apply_mutation(self, sequence_object):
        # 1. Load the Likelihoods File
        if not self.loaded_metadata:
            self.load_metadata(sequence_object)
            self.loaded_metadata = True

        # 1.1 Sample Mutation Rate
        mutation_rate = random.uniform(self.min_mutation_rate, self.max_mutation_rate)
        target_number_of_mutations = int(mutation_rate * len(sequence_object.ungapped_seq))

        # have a copy of the original neucliotdies
        naive_sequence = list(sequence_object.ungapped_seq)

        # Log mutations
        mutations = dict()

        # 2. Extract 5-Mers
        fiver_mers = FiveMER.create_five_mers(sequence_object.ungapped_seq, self.mutability)

        # add a failsafe to insure while loop does not get locked
        patience = 0

        while len(mutations) < target_number_of_mutations:
            # 3. Mutability, Weighted Choice of Position Based on 5-Mer Likelihoods
            sampled_position = self.weighted_choice(fiver_mers)  # likelihoods are normalized here

            # 4. Substitution
            substitutions = self.substitution[sampled_position.sequence].dropna()  # drop Nan's - N's and Same Base
            mutable_bases = substitutions.index
            bases_likelihoods = substitutions.values
            mutation_to_apply = random.choices(mutable_bases, weights=bases_likelihoods, k=1)[0]

            # log
            if sampled_position.position not in mutations:
                mutations[sampled_position.position] = f'{sampled_position.sequence[2]}>{mutation_to_apply}'
            else:
                mutations[sampled_position.position] += f'>{mutation_to_apply}'

            # if mutation reverted previous mutation back to naive state, drop that record from the log
            if mutation_to_apply == naive_sequence[sampled_position.position]:
                mutations.pop(sampled_position.position)

            # 5. Apply Mutation
            # This will also update all relevant 5-MERS and their likelihood with pointer like logic
            sampled_position.change_center(mutation_to_apply, self.mutability)

            patience += 1
            # Patience logic
            if patience > (target_number_of_mutations * 30) and target_number_of_mutations > 1:
                patience = 0
                # restart process
                mutation_rate = random.uniform(self.min_mutation_rate, self.max_mutation_rate)
                target_number_of_mutations = int(mutation_rate * len(sequence_object.ungapped_seq))
                mutations = dict()
                # 2. Extract 5-Mers
                fiver_mers = FiveMER.create_five_mers(sequence_object.ungapped_seq, self.mutability)

        mutated_sequence = FiveMER.five_mers_to_dna(fiver_mers)
        return mutated_sequence, mutations, mutation_rate

    def _mutate_base(self, base):
        return random.choice(list(self.bases - {base}))

    @staticmethod
    def weighted_choice(five_mers):
        weights = [fm.likelihood if fm.likelihood == fm.likelihood else 0 for fm in five_mers]
        # Choose an index instead of the object
        chosen_index = random.choices(range(len(five_mers)), weights, k=1)[0]
        return five_mers[chosen_index]
