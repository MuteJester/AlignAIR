import numpy as np


class FastKmerDensityExtractor:
    def __init__(self, k, max_length, allowed_mismatches=0):
        self.k = k
        self.max_length = max_length
        self.allowed_mismatches = allowed_mismatches
        self.kmer_set = set()

    def create_kmers(self, reference_sequences):
        def generate_variants(kmer, mismatches):
            if mismatches == 0:
                return [kmer]
            variants = []
            for i in range(len(kmer)):
                for char in 'ACGT':  # Assuming DNA/RNA sequence
                    if char != kmer[i]:
                        new_variant = kmer[:i] + char + kmer[i + 1:]
                        variants.extend(generate_variants(new_variant, mismatches - 1))
            return set(variants)

        for seq in reference_sequences:
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i + self.k]
                if self.allowed_mismatches > 0:
                    self.kmer_set.update(generate_variants(kmer, self.allowed_mismatches))
                else:
                    self.kmer_set.add(kmer)

    def fit(self, reference_sequences):
        self.create_kmers(reference_sequences)

    def transform_holt(self, sequence, alpha=0.05, beta=0.01):
        sequence_length = len(sequence)
        max_density = 0
        max_region_start = 0

        # Calculate the initial window
        current_density = 0
        for i in range(min(self.max_length, sequence_length)):
            if sequence[i:i + self.k] in self.kmer_set:
                current_density += 1

        # Initialize the Holt's linear method parameters
        level = current_density
        trend = 0
        holt = level + trend
        max_density = holt
        max_region_start = 0

        # Slide the window over the sequence
        d_history = [holt]

        for i in range(1, sequence_length - self.max_length + 1):
            if sequence[i - 1:i - 1 + self.k] in self.kmer_set:
                current_density -= 1
            if sequence[i + self.max_length - self.k:i + self.max_length] in self.kmer_set:
                current_density += 1

            prev_level = level
            level = alpha * current_density + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            holt = level + trend

            if holt >= max_density:
                max_density = holt
                max_region_start = i

            d_history.append(holt)

        max_region = sequence[max_region_start:max_region_start + self.max_length]
        return max_region, d_history
