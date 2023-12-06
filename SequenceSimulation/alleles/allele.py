import re
import random
from abc import ABC, abstractmethod
from SequenceSimulation.utilities import weighted_choice, TrimMode
from enum import Enum, auto

class AlleleTypes(Enum):
    V = auto()
    D = auto()
    J = auto()

class Allele(ABC):
    def __init__(self, name, gapped_seq, length):
        self.name = name
        self.gapped_seq = gapped_seq
        self.length = int(length)
        self.ungapped_seq = gapped_seq.replace(".", "")
        self.ungapped_len = len(self.ungapped_seq)
        self.family = self.name.split("-")[0] if "-" in self.name else self.name.split("*")[0]
        self.gene = self.name.split("*")[0]
        self.anchor = None
        self._find_anchor()

    @abstractmethod
    def _find_anchor(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.gapped_seq[:10]}, {self.length})"

    @abstractmethod
    def _get_trim_length(self, trim_dicts,trim_mode=TrimMode.DEFAULT ):
        """Generate trimmed ungapped allele nucleotide sequence.

        Args:
            allele (Allele): Allele to be trimmed.
            no_trim_list (list): List of 5 Booleans, specifying whether to not
                trim [all_ends, v_3_end, d_5_end, d_3_end, j_5_end].
            trim_dicts (dict): A dictionary of dictionaries of trimming length
                proportions by gene family for each segment (V, D or J).

        Returns:
            trimmed_seq (str): Trimmed nucleotide sequence, lower case, ungapped.
        """

        pass

    @abstractmethod
    def get_trimmed(self,trim_dict,trim_mode=TrimMode.DEFAULT,gapped=False):
        pass

class VAllele(Allele):
    type = AlleleTypes.V

    def _find_anchor(self):
        cys_wider = self.gapped_seq[306:315]
        self.anchor = self.ungapped_seq.rfind(cys_wider) + 3

    def _get_trim_length(self, trim_dicts,trim_mode=TrimMode.DEFAULT ):
        trim_3 = 0  # set to 0 - J will never be trimmed at 3'
        trim_5 = 0  # set to 0 - V will never be trimmed at 5'

        if trim_mode == TrimMode.DEFAULT:
            trim_3_dict = trim_dicts["V_3"]
            # choose trim length/prob dict by gene family
            if self.family in trim_3_dict:
                prob_dict = trim_3_dict[self.family]
            else:
                print(self.family, 'NOT IN TRIM PROBABILITY DICTIONARY!, CHOOSING RANDOM!')
                prob_dict = random.choice(list(trim_3_dict.values()))

            # prevent entire allele or anchor from being removed
            valid_trim_amounts = filter(lambda amount: (amount < self.length) or \
                                                       (amount < (self.length - self.anchor - 1)), prob_dict)

            prob_dict = {amount: prob_dict[amount] for amount in valid_trim_amounts}

            trim_3 = weighted_choice(prob_dict)

        return trim_5, trim_3

    def get_trimmed(self,trim_dict,trim_mode=TrimMode.DEFAULT,gapped=False):
        sequence = self.gapped_seq if gapped else self.ungapped_seq
        trim_5, trim_3 = self._get_trim_length(trim_dict, trim_mode)
        trimmed_seq = sequence[:-trim_3 if trim_3 > 0 else None]
        return trimmed_seq,trim_5, trim_3

class DAllele(Allele):
    type = AlleleTypes.D

    def _find_anchor(self):
        # D alleles might not have a specific anchor finding logic,
        # implement their specific behavior here if needed
        pass

    def _get_trim_length(self, trim_dicts,trim_mode=TrimMode.DEFAULT ):
        trim_3 = 0  # set to 0 - J will never be trimmed at 3'
        trim_5 = 0  # set to 0 - V will never be trimmed at 5'

        if trim_mode != TrimMode.NO_5_PRIME:
            trim_5_dict = trim_dicts["D_5"]
            if self.family in trim_5_dict:
                prob_5_dict = trim_5_dict[self.family]
            else:
                prob_5_dict = random.choice(list(trim_5_dict.values()))
                print(self.family, 'NOT IN TRIM PROBABILITY DICTIONARY!, CHOOSING RANDOM!')
            trim_5 = weighted_choice(prob_5_dict)

        if trim_mode != TrimMode.NO_3_PRIME:
            trim_3_dict = trim_dicts["D_3"]
            if self.family in trim_3_dict:
                prob_3_dict = trim_3_dict[self.family]
            else:
                prob_3_dict = random.choice(list(trim_3_dict.values()))
                print(self.family, 'NOT IN TRIM PROBABILITY DICTIONARY!, CHOOSING RANDOM!')

            valid_d3_trim_amounts = filter(lambda amount: amount + trim_5 < self.length, prob_3_dict)

            prob_3_dict = {amount: prob_3_dict[amount] for amount in valid_d3_trim_amounts}

            trim_3 = weighted_choice(prob_3_dict)
        return trim_5,trim_3

    def get_trimmed(self, trim_dict, trim_mode=TrimMode.DEFAULT,gapped=False):
        sequence = self.gapped_seq if gapped else self.ungapped_seq
        trim_5, trim_3 = self._get_trim_length(trim_dict,trim_mode)
        trimmed_seq = sequence[trim_5:len(sequence) - trim_3 if trim_3 > 0 else None]
        return trimmed_seq,trim_5, trim_3

class JAllele(Allele):
    type = AlleleTypes.J

    def _find_anchor(self):
        motif = re.compile('(ttt|ttc|tgg)(ggt|ggc|gga|ggg)[a-z]{3}(ggt|ggc|gga|ggg)')
        match = motif.search(self.ungapped_seq)
        self.anchor = match.span()[0] if match else None

    def _get_trim_length(self, trim_dicts,trim_mode=TrimMode.DEFAULT ):
        trim_3 = 0  # set to 0 - J will never be trimmed at 3'
        trim_5 = 0  # set to 0 - V will never be trimmed at 5'

        if trim_mode != TrimMode.NO_5_PRIME:
            trim_5_dict = trim_dicts["J_5"]
            if self.family in trim_5_dict:
                prob_dict = trim_5_dict[self.family]
            else:
                prob_dict = random.choice(list(trim_5_dict.values()))
                print(self.family, 'NOT IN TRIM PROBABILITY DICTIONARY!, CHOOSING RANDOM!')

            valid_5_trims = filter(lambda t5: (t5 < self.length) or (t5 < self.anchor),prob_dict)
            prob_dict = {amount:prob_dict[amount] for amount in valid_5_trims}
            trim_5 = weighted_choice(prob_dict)
        return trim_5,trim_3

    def get_trimmed(self,trim_dict,trim_mode=TrimMode.DEFAULT,gapped=False):
        trim_5, trim_3 = self._get_trim_length(trim_dict, trim_mode)
        sequence = self.gapped_seq if gapped else self.ungapped_seq
        return sequence[trim_5:],trim_5, trim_3

