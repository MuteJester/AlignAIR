import random

from SequenceSimulation.alleles import AlleleTypes
from abc import ABC, abstractmethod

from SequenceSimulation.sequence import NP_Region
from SequenceSimulation.utilities import NP
from SequenceSimulation.utilities.data_config import DataConfig


class BaseSequence(ABC):
    """
    Represents a recombined Ig sequence consisting of V, D and J segments.

    Attributes:
        v_allele (Allele): IMGT V gene allele.
        d_allele (Allele): IMGT D gene allele.
        j_allele (Allele): IMGT J gene allele.
        NP1_region (str): NP1 region - between V and D gene.
        NP1_length (int): Length of NP1 region.
        NP2_region (str): NP2 region - between V and D gene.
        NP2_length (int): Length of NP2 region.
        ungapped_seq (str): Ungapped nucleotide sequence.
        gapped_seq (str): Gapped nucleotide sequence.
        mutated_seq (str): Ungapped mutated nucleotide sequence.
        gapped_mutated_seq (str): Ungapped mutated nucleotide sequence.
        mutated_seq (str): Ungapped mutated nucleotide sequence.
        junction (str): Nucleotide sequence of junction region.
        v_seq (str):  Nucleotide sequence of V region.
        d_seq (str): Nucleotide sequence of D region.
        j_seq (str): Nucleotide sequence of J region.
        v_seq_start (int): Start position of V region.
        d_seq_start (int): Start position of D region.
        j_seq_start (int): Start position of J region.
        v_seq_end (int): End position of V region.
        d_seq_end (int): End position of D region.
        j_seq_end (int): End position of J region.
        mutations (str): Mutation events.
        mut_count (int): Mutation count.
        mut_freq (int): Mutation frequency.
    """

    def __init__(self, alleles):
        """Initialises a Sequence class instance.

        Args:
            alleles List[(Allele)]: IMGT V/D/J gene allele, required At least 2 (V/J) or (V/D/J).
        """
        self.v_allele = next((allele for allele in alleles if allele.type == AlleleTypes.V), None)
        assert self.v_allele is not None  # Must Have V Allele!
        self.d_allele = next((allele for allele in alleles if allele.type == AlleleTypes.D), None)
        self.j_allele = next((allele for allele in alleles if allele.type == AlleleTypes.J), None)
        assert self.j_allele is not None  # Must Have J Allele!
        self.NP1_region = ""
        self.NP2_region = ""
        self.NP1_length = 0
        self.NP2_length = 0
        self.junction = ""
        self.v_seq = ""
        self.d_seq = ""
        self.j_seq = ""
        self.v_seq_start = 0
        self.d_seq_start = 0
        self.j_seq_start = 0
        self.v_seq_end = 0
        self.d_seq_end = 0
        self.j_seq_end = 0
        self.mutations = ""
        self.mut_count = 0
        self.mut_freq = 0
        self.ungapped_seq = ""
        self.mutated_seq = None
        self.gapped_seq = ""
        self.gapped_mutated_seq = None

    @abstractmethod
    def simulate_sequence(self, dataconfig: DataConfig):

        """Creates the recombined nucleotide sequence with trimming and np addition.

                Args:
                    no_trim_list (list): List of 5 Booleans, specifying whether to not
                        trim [all_ends, v_3_end, d_5_end, d_3_end, j_5_end].
                    trim_dicts (dict): A dictionary of dictionaries of trimming length
                        proportions by gene family for each segment (V, D or J).
                    no_np_list (list): List of 3 Booleans, specifying whether to not
                        add [both_np, np1, np2].
                    NP_lengths (dict): Dictionary of possible NP region lengths and the
                        proportion of sequences to use them. In the format
                        {NP region length: proportion}.
                    NP_transitions (dict): Nested dictionary containing transition matrix of
                        probabilities of moving from one nucleotide (A, C, G, T) to any other
                        for each position in the NP region.
                    NP_first_bases (dict): Nested dictionary of the proportion of NP
                        sequences starting with each base for NP1 and NP2.
                    gapped (bool): Specify whether to return sequence with IMGT gaps
                        or not.

                Returns:
                    nuc_seq (str): The recombined nucleotide sequence.
                """

        pass

    @abstractmethod
    def get_junction_length(self):
        """Calculates the junction length of the sequence (CDR3 region plus both
                anchor residues).

                Returns:
                    junction_length (int): Number of nucleotides in junction (CDR3 + anchors)
        """
        pass


class HeavyChainSequence(BaseSequence):

    def __init__(self,alleles,dataconfig: DataConfig):
        super().__init__(alleles)
        self.gapped_seq = self.simulate_sequence(dataconfig)
        self.ungapped_seq = self.gapped_seq.replace(".", "")
        self.junction_length = self.get_junction_length()


    def simulate_sequence(self, dataconfig: DataConfig):

        #trim_dicts, np_usage, NP_lengths, NP_transitions, NP_first_bases,
                    #gapped=False, trim_modes=None

        v_allele = self.v_allele
        d_allele = self.d_allele
        j_allele = self.j_allele

        if dataconfig.np_usage != NP.NONE:
            if dataconfig.np_usage == NP.NP1_ONLY or dataconfig.np_usage == NP.NP1_NP2:
                self.NP1_region = NP_Region.create_np_region(dataconfig.NP_lengths, dataconfig.NP_transitions, "NP1", dataconfig.NP_first_bases)
            if dataconfig.np_usage != NP.NP2_ONLY or dataconfig.np_usage == NP.NP1_NP2:
                self.NP2_region = NP_Region.create_np_region(dataconfig.NP_lengths, dataconfig.NP_transitions, "NP2", dataconfig.NP_first_bases)

        self.NP1_length = len(self.NP1_region)
        self.NP2_length = len(self.NP2_region)

        v_trimmed_seq, v_trim_5, v_trim_3 = v_allele.get_trimmed(dataconfig.trim_dicts, dataconfig.trim_modes['V'], dataconfig.gapped)
        d_trimmed_seq, d_trim_5, d_trim_3 = d_allele.get_trimmed(dataconfig.trim_dicts, dataconfig.trim_modes['D'], dataconfig.gapped)
        j_trimmed_seq, j_trim_5, j_trim_3 = j_allele.get_trimmed(dataconfig.trim_dicts, dataconfig.trim_modes['J'], dataconfig.gapped)

        nuc_seq = (
                v_trimmed_seq
                + self.NP1_region
                + d_trimmed_seq
                + self.NP2_region
                + j_trimmed_seq
        )

        # log trims
        self.v_trim_5 = v_trim_5
        self.v_trim_3 = v_trim_3
        self.d_trim_5 = d_trim_5
        self.d_trim_3 = d_trim_3
        self.j_trim_5 = j_trim_5
        self.j_trim_3 = j_trim_3

        return nuc_seq.upper()

    def get_junction_length(self):
        """Calculates the junction length of the sequence (CDR3 region plus both
        anchor residues).

        Returns:
            junction_length (int): Number of nucleotides in junction (CDR3 + anchors)
        """

        junction_length = self.v_allele.length - (
                self.v_allele.anchor - 1) - self.v_trim_3 + self.NP1_length + self.d_allele.length - \
                          self.d_trim_5 - self.d_trim_3 + \
                          self.NP2_length + (self.j_allele.anchor + 2) - self.j_trim_5
        return junction_length

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
        random_d_allele = random.choice([i for j in dataconfig.d_alleles for i in dataconfig.d_alleles[j]])
        random_j_allele = random.choice([i for j in dataconfig.j_alleles for i in dataconfig.j_alleles[j]])
        return cls([random_v_allele, random_d_allele, random_j_allele],dataconfig)
