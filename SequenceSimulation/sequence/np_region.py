from SequenceSimulation.utilities import weighted_choice


class NP_Region:
    """
    Class that represents an NP region where base usage is determined by a first
    order Markov chain for which the transition matrix varies by position.

    Attributes:
        transition_probs (dict) : Dictionary of transition matrices per position
            in sequence.
        first_base (str) : The first base in the NP region.
        length (int) : The chosen length of the NP region.
    """

    def __init__(self, transition_probs, first_base, length):
        """Initialises an NP_region class instance.
        """

        self.transition_probs = transition_probs
        self.bases = ["A", "C", "G", "T"]
        self.first_base = first_base
        self.length = length

    def next_base(self, current_base, position):
        """Get the next base in NP region from transition matrix.

        Args:
            current_base (str): Current base in NP region sequence.
            position (int): Current position in NP region sequence.

        Returns:
            base (str): Next base in the NP region sequence.
        """

        base = weighted_choice(
            {next_base: self.transition_probs[position][current_base][next_base] for next_base in \
             list(self.transition_probs[position][current_base])})
        return base

    def generate_np_seq(self):
        """Creates an NP region sequence using a first order Markov chain.

        Returns:
            sequence (str): Final NP region sequence.
        """
        sequence = ""
        current_base = self.first_base
        sequence += current_base
        for i in range(self.length - 1):
            next_base = self.next_base(current_base, i)
            sequence += next_base
            current_base = next_base
        return sequence.lower()

    @classmethod
    def create_np_region(cls, NP_lengths, NP_transitions, which_NP, first_base_dict):
        length = weighted_choice(NP_lengths[which_NP])
        if length > 0:
            first_base = weighted_choice(first_base_dict[which_NP])
            np_region = cls(NP_transitions[which_NP], first_base, length)
            return np_region.generate_np_seq()
        return ""
