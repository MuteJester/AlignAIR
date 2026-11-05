"""Allele multi-hot encoding of truth calls against a reference's allele order."""
import numpy as np

from ..errors import TrainingDataError


def encode_allele_sets(allele_sets, allele_to_index, count: int, *, strict=False) -> np.ndarray:
    """Encode against a fixed reference order; strict training adapters reject unknown truth labels."""
    allele_sets = list(allele_sets)
    encoded = np.zeros((len(allele_sets), count), dtype=np.float32)
    for row, names in enumerate(allele_sets):
        for name in names:
            position = allele_to_index.get(name)
            if position is None:
                if strict:
                    raise TrainingDataError("training allele is not in the model reference",
                                            context={"allele": name})
            else:
                encoded[row, position] = 1
    return encoded

