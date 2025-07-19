from typing import List, Union
import numpy as np
from GenAIRR.dataconfig.enums import ChainType


class ChainTypeOneHotEncoder:
    """
    A simple one-hot encoder for chain types.
    """

    def __init__(self, chain_types: List[ChainType]):
        self.chain_types = chain_types
        # sort chain types to ensure consistent ordering
        chain_types.sort(key=lambda x: x.value)  # Sort by the enum value
        self.type_to_index = {chain_type: i for i, chain_type in enumerate(chain_types)}
        self.index_to_type = {i: chain_type for i, chain_type in enumerate(chain_types)}

    def _encode_single(self,chain_type):
        index = self.type_to_index.get(chain_type, -1)
        if index == -1:
            raise ValueError(f"Chain type '{chain_type}' not recognized.")
        one_hot_vector = [0] * len(self.chain_types)
        one_hot_vector[index] = 1
        one_hot_vector = np.array(one_hot_vector)
        return one_hot_vector

    def encode(self, chain_type):
        """
        Encode a chain type as a one-hot vector.
        """
        if isinstance(chain_type, list):
            return np.vstack([self._encode_single(ct) for ct in chain_type])
        elif isinstance(chain_type, ChainType):
            return self._encode_single(chain_type)
        else:
            raise ValueError("chain_type must be a ChainType instance or a list of ChainType instances.")

    def _decode_single(self, one_hot_vector):
        """
        Decode a single one-hot vector back to a chain type.
        """
        if len(one_hot_vector) != len(self.chain_types):
            raise ValueError("One-hot vector length does not match number of chain types.")
        index = np.argmax(one_hot_vector)
        return self.index_to_type[index]

    def decode(self, one_hot_vector):
        """
        Decode a one-hot vector back to a chain type.
        """
        if one_hot_vector.shape[1] != len(self.chain_types):
            raise ValueError("One-hot vector length does not match number of chain types.")

        decoded = []
        for i in range(one_hot_vector.shape[0]):
            decoded.append(self._decode_single(one_hot_vector[i]))
        return decoded

    def __repr__(self):
        return f"ChainTypeOneHotEncoder(chain_types={self.chain_types})"