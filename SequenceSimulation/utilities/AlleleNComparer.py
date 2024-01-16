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

        # Iterate through all alleles and compare them at non-N positions
        for other_allele_id, sequence in self.alleles.items():
            is_indistinguishable = True

            # Only compare up to the length of the shorter allele
            for pos in range(min(len(target_sequence), len(sequence))):
                # Skip comparison at 'N' positions
                if pos in n_positions:
                    continue

                # If nucleotides differ at any non-'N' position, alleles are distinguishable
                if target_sequence[pos] != sequence[pos]:
                    is_indistinguishable = False
                    break

            if is_indistinguishable:
                indistinguishable_alleles.add(other_allele_id)

        return indistinguishable_alleles