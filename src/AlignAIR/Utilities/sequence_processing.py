from multiprocessing import cpu_count, Pool

import numpy as np
from tqdm.auto import tqdm


def encode_and_equal_pad_sequence(sequence, max_seq_length, tokenizer_dictionary):
    """Encodes a sequence of nucleotides and pads it to the specified maximum length, equally from both sides.

    Args:
        sequence: A sequence of nucleotides.

    Returns:
        A padded sequence, and the start and end indices of the unpadded sequence.
    """

    encoded_sequence = np.array([tokenizer_dictionary[i] for i in sequence], dtype=np.int32)
    padding_length = max_seq_length - len(encoded_sequence)
    iseven = padding_length % 2 == 0
    pad_size = padding_length // 2
    if iseven:
        encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size), 'constant', constant_values=(0, 0))
    else:
        encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size + 1), 'constant', constant_values=(0, 0))
    return encoded_sequence


def tokenize_chunk(chunk, max_seq_length, tokenizer_dictionary):
    return [(index, encode_and_equal_pad_sequence(sequence, max_seq_length, tokenizer_dictionary)) for index, sequence
            in chunk]


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def tokenize_sequences(sequences, max_seq_length, tokenizer_dictionary, verbose=False):
    num_cpus = cpu_count()
    indexed_sequences = list(enumerate(sequences))
    chunks = chunkify(indexed_sequences, num_cpus)

    # Create a partial function that includes the fixed arguments
    from functools import partial
    tokenize_partial = partial(tokenize_chunk, max_seq_length=max_seq_length, tokenizer_dictionary=tokenizer_dictionary)

    with Pool(num_cpus) as pool:
        if verbose:
            results = list(tqdm(pool.imap(tokenize_partial, chunks), total=len(chunks)))
        else:
            results = pool.map(tokenize_partial, chunks)

    # Flatten the list of lists and sort by the original index to maintain order
    tokenized_sequences = [seq for chunk in results for seq in chunk]
    tokenized_sequences.sort(key=lambda x: x[0])

    # Remove the indices and extract the tokenized sequences
    tokenized_sequences = [seq for index, seq in tokenized_sequences]
    return np.vstack(tokenized_sequences)


def tokenize_sequences_batch(sequences, max_seq_length, tokenizer_dictionary):
    tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in
                           sequences]
    return np.vstack(tokenized_sequences).astype(np.int32)

