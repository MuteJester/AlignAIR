import numpy as np

class CenterPaddedSequenceTokenizer:
    """
    Encodes and center-pads nucleotide sequences using a token dictionary.

    This tokenizer ensures that the original sequence appears in the center
    of the final array, with equal (or nearly equal) padding on both sides.

    Attributes:
        token_dict (dict): Mapping of nucleotides to integers.
        max_length (int): Maximum padded sequence length.
    """

    def __init__(self, token_dict=None, max_length=576):
        """
        Args:
            token_dict (dict, optional): Custom token mapping. Defaults to standard A/T/G/C/N/P.
            max_length (int): Desired padded sequence length.
        """
        self.token_dict = token_dict or {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0  # Padding token
        }
        self.max_length = max_length

    def encode(self, sequence):
        """
        Converts nucleotide string to an encoded array of integers.

        Args:
            sequence (str): Input nucleotide string.

        Returns:
            np.ndarray: Encoded sequence.
        """
        # Emit int32 token ids for compatibility with Keras Embedding and SavedModel signature
        return np.array([self.token_dict.get(nt, self.token_dict['N']) for nt in sequence], dtype=np.int32)

    def encode_and_pad_center(self, sequence_or_sequences):
        """
        Encodes and center-pads one or multiple nucleotide sequences.

        Args:
            sequence_or_sequences (str or Iterable[str]):
                A single nucleotide string or an iterable of nucleotide strings.

        Returns:
            If input is a string:
                Tuple[np.ndarray, int]: (padded_sequence, left_padding)
            If input is an iterable:
                Tuple[np.ndarray, np.ndarray]: (batch_padded_sequences, paddings_array)
        """
        if isinstance(sequence_or_sequences, str):
            encoded = self.encode(sequence_or_sequences)
            padding_length = self.max_length - len(encoded)
            pad_left = padding_length // 2
            pad_right = padding_length - pad_left
            padded = np.pad(encoded, (pad_left, pad_right), constant_values=0).astype(np.int32, copy=False)
            return padded, pad_left

        # If it's a list/iterable of sequences
        padded_batch = []
        paddings = []

        for seq in sequence_or_sequences:
            encoded = self.encode(seq)
            padding_length = self.max_length - len(encoded)
            pad_left = padding_length // 2
            pad_right = padding_length - pad_left
            padded = np.pad(encoded, (pad_left, pad_right), constant_values=0).astype(np.int32, copy=False)
            padded_batch.append(padded)
            paddings.append(pad_left)

        return np.vstack(padded_batch).astype(np.int32, copy=False), np.array(paddings, dtype=np.int32)


