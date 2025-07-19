from dataclasses import dataclass
import numpy as np

@dataclass
class GeneEncoding:
    allele_to_index: dict
    index_to_allele: dict
    count: int

class AlleleEncoder:
    """
    Handles one-hot encoding and reverse mapping of allele calls for each gene type.
    """

    def __init__(self):
        self.gene_encodings = {}  # e.g., {"V": GeneEncoding(...), ...}

    def register_gene(self, gene_type, allele_list, sort=True, allow_overwrite=False):
        """
        Registers alleles for a gene type.

        Args:
            gene_type (str): Gene identifier (e.g., "V", "D", "J").
            allele_list (list[str]): Alleles to register.
            sort (bool): Whether to sort the list before mapping.
            allow_overwrite (bool): Allow overwriting an existing gene entry.
        """
        if gene_type in self.gene_encodings and not allow_overwrite:
            raise ValueError(f"Gene type '{gene_type}' is already registered. Use allow_overwrite=True to replace it.")

        alleles = sorted(allele_list) if sort else allele_list
        allele_to_index = {allele: idx for idx, allele in enumerate(alleles)}
        index_to_allele = {idx: allele for allele, idx in allele_to_index.items()}
        self.gene_encodings[gene_type] = GeneEncoding(allele_to_index, index_to_allele, len(alleles))

    def encode(self, gene_type, allele_sets):
        """
        One-hot encodes a list of allele sets for a specific gene.

        Args:
            gene_type (str): "V", "D", or "J"
            allele_sets (Iterable[set[str]]): Each set may contain multiple true alleles.

        Returns:
            np.ndarray: (n_samples, n_alleles) one-hot encoded array.
        """
        enc = self.gene_encodings[gene_type]
        result = []

        for sample in allele_sets:
            ohe = np.zeros(enc.count, dtype=np.float32)
            for allele in sample:
                idx = enc.allele_to_index.get(allele)
                if idx is not None:
                    ohe[idx] = 1
            result.append(ohe)

        return np.vstack(result)

    def get_reverse_mapping(self):
        """
        Returns: dict[str, dict[int, str]] for each gene type.
        """
        return {gene: enc.index_to_allele for gene, enc in self.gene_encodings.items()}

    def get_properties_map(self):
        """
        Returns: dict[str, dict[str, Any]] — for DatasetBase compatibility
        """
        return {
            gene: {
                "allele_count": enc.count,
                "allele_call_ohe": enc.allele_to_index,
                'reverse_mapping': enc.index_to_allele
            }
            for gene, enc in self.gene_encodings.items()
        }
