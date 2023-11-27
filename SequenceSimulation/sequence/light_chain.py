import random

from SequenceSimulation.mutation import MutationModel
from SequenceSimulation.sequence import NP_Region
from SequenceSimulation.sequence.sequence import BaseSequence
from SequenceSimulation.utilities import NP, translate
from SequenceSimulation.utilities.data_config import DataConfig


class LightChainSequence(BaseSequence):

    def __init__(self, alleles, dataconfig: DataConfig):
        super().__init__(alleles)
        self.simulate_sequence(dataconfig)

    def simulate_sequence(self, dataconfig: DataConfig):
        v_allele = self.v_allele
        j_allele = self.j_allele

        if dataconfig.np_usage == NP.DEFAULT:
            self.NP1_region = NP_Region.create_np_region(dataconfig.NP_lengths, dataconfig.NP_transitions, "NP1",
                                                         dataconfig.NP_first_bases)

        self.NP1_length = len(self.NP1_region)

        v_trimmed_seq, v_trim_5, v_trim_3 = v_allele.get_trimmed(dataconfig.trim_dicts, dataconfig.trim_modes['V'],
                                                                 dataconfig.gapped)

        j_trimmed_seq, j_trim_5, j_trim_3 = j_allele.get_trimmed(dataconfig.trim_dicts, dataconfig.trim_modes['J'],
                                                                 dataconfig.gapped)

        nuc_seq = (
                v_trimmed_seq +
                self.NP1_region +
                j_trimmed_seq
        )

        # log trims
        self.v_trim_5 = v_trim_5
        self.v_trim_3 = v_trim_3
        self.j_trim_5 = j_trim_5
        self.j_trim_3 = j_trim_3

        self.gapped_seq = nuc_seq.upper()
        self.ungapped_seq = self.gapped_seq.replace(".", "")

        self.junction_length = self.get_junction_length()

        self.junction = self.ungapped_seq[self.v_allele.anchor:
                                          self.v_allele.anchor + self.junction_length].upper()

        self.update_metadata()
        self._is_functional(self.ungapped_seq)

    def update_metadata(self):

        self.v_seq_start = 0
        self.v_seq_end = self.v_allele.ungapped_len - self.v_trim_3
        self.j_seq_start = self.v_seq_end + self.NP1_length
        self.j_seq_end = self.j_seq_start + self.j_allele.ungapped_len - self.j_trim_5 - 1

    def _is_functional(self, sequence):
        self.functional = False
        if (self.junction_length % 3) == 0 and self.check_stops(sequence) is False:
            self.junction_aa = translate(self.junction)
            if self.junction_aa.startswith("C") and (
                    self.junction_aa.endswith("F") or self.junction_aa.endswith("W")):
                self.functional = True

    def mutate(self, mutation_model: MutationModel):
        mutated_sequence, mutations, mutation_rate = mutation_model.apply_mutation(self)
        self.mutated_seq = mutated_sequence
        self.mutations = mutations
        self.mutation_freq = mutation_rate
        self.mutation_count = len(mutations)

        self.junction = self.mutated_seq[self.v_allele.anchor:
                                         self.v_allele.anchor + self.junction_length].upper()
        # mutation metadata updates
        self._is_functional(self.mutated_seq)

    def get_junction_length(self):
        """Calculates the junction length of the sequence (CDR3 region plus both
        anchor residues).

        Returns:
            junction_length (int): Number of nucleotides in junction (CDR3 + anchors)
        """

        junction_length = self.v_seq_end - (self.v_allele.anchor - 1) - self.v_trim_3 + self.NP1_length + \
                          (self.j_allele.anchor + 2) - self.j_trim_5
        return junction_length

    def check_stops(self, seq):
        """Check for stop codons in a nucleotide sequence.

        Args:
            sequence (str): Nucleotide sequence.

        Returns:
            stop (bool): True if stop codon is present.
        """
        stops = ["TAG", "TAA", "TGA"]
        for x in range(0, len(seq), 3):
            if seq[x:x + 3] in stops:
                return True
        return False

    @classmethod
    def create_random(cls, dataconfig: DataConfig):
        """
        Creates a random instance of HeavyChainSequence with a random V, D, and J allele.

        Args:
            dataconfig (Type[DataConfig]): DataConfig Object with Allele Infomration

        Returns:
            HeavyChainSequence: An instance of HeavyChainSequence with randomly selected alleles.
        """
        random_v_allele = random.choice([i for j in dataconfig.v_alleles for i in dataconfig.v_alleles[j]])
        random_j_allele = random.choice([i for j in dataconfig.j_alleles for i in dataconfig.j_alleles[j]])
        return cls([random_v_allele, random_j_allele], dataconfig)

    def __repr__(self):
        # Calculate proportional lengths for the drawing
        total_length = self.j_seq_end  # Assuming j_seq_end is the end of the sequence
        proportional_length = lambda start, end: int((end - start) / total_length * 100)  # Example scale factor

        v_length = proportional_length(self.v_seq_start, self.v_seq_end)
        j_length = proportional_length(self.j_seq_start, self.j_seq_end)

        # Construct ASCII drawing
        v_part = f"{self.v_seq_start}|{'-' * v_length}V({self.v_allele.name})|{self.v_seq_end}"
        j_part = f"{self.j_seq_start}|{'-' * j_length}J({self.j_allele.name})|{self.j_seq_end}"

        return f"{v_part}{j_part}"
