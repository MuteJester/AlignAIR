import pickle

class AlleleNComparer:
    def __init__(self):
        self.alleles = {}  # Stores the sequences of each allele

    def add_allele(self, allele_id, sequence):
        self.alleles[allele_id] = sequence

    def find_indistinguishable_alleles(self, allele_id, n_positions):
        if allele_id not in self.alleles:
            return f"Allele {allele_id} not found."

        target_sequence = self.alleles[allele_id]
        indistinguishable_alleles = set()

        for other_allele_id, sequence in self.alleles.items():
            is_indistinguishable = True
            for pos in range(min(len(target_sequence), len(sequence))):
                if pos in n_positions:
                    continue
                if target_sequence[pos] != sequence[pos]:
                    is_indistinguishable = False
                    break

            if is_indistinguishable:
                indistinguishable_alleles.add(other_allele_id)

        return indistinguishable_alleles

    def save(self, filename):
        """Saves the alleles dictionary to a file using pickle."""
        with open(filename, 'wb') as file:
            pickle.dump(self.alleles, file)

    def load(self, filename):
        """Loads the alleles dictionary from a file using pickle."""
        with open(filename, 'rb') as file:
            self.alleles = pickle.load(file)

