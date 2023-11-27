import random
from SequenceSimulation.mutation.mutation_model import MutationModel


class Uniform(MutationModel):
    def __init__(self, min_mutation_rate=0, max_mutation_rate=0):
        self.max_mutation_rate = max_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.bases = {'A','T','C','G'}


    def mutable_positions(self,sequence):
        """
        this function extract all positions in a sequence ignoring NP regions
        :return:
        """

        positions_to_mutate = []

        # add v region positions
        positions_to_mutate += list(range(sequence.v_seq_start,sequence.v_seq_end))
        positions_to_mutate += list(range(sequence.d_seq_start,sequence.d_seq_end))
        positions_to_mutate += list(range(sequence.j_seq_start,sequence.j_seq_end))

        return positions_to_mutate

    def apply_mutation(self, sequence_object):
        sequence = sequence_object.ungapped_seq
        mutation_rate = random.uniform(self.min_mutation_rate, self.max_mutation_rate)
        number_of_mutations = int(mutation_rate*len(sequence))
        positions_to_mutate = self.mutable_positions(sequence_object)
        positions_to_mutate = random.choices(positions_to_mutate,k=number_of_mutations)

        # log mutations
        mutations = dict()
        mutated_sequence = list(sequence)

        for position in positions_to_mutate:
            new_base = self._mutate_base(mutated_sequence[position])
            mutations[position] = f'{mutated_sequence[position]}>{new_base}'
            mutated_sequence[position] = new_base

        mutated_sequence = ''.join(mutated_sequence)
        return mutated_sequence, mutations, mutation_rate

    def _mutate_base(self, base):
        return random.choice(list(self.bases - {base}))
