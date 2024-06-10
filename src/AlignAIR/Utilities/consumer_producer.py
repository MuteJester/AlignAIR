import numpy as np
import pandas as pd

from AlignAIR.Utilities.sequence_processing import encode_and_equal_pad_sequence


def sequence_tokenizer_worker(file_path, queue, max_seq_length, tokenizer_dictionary, batch_size=256):
    sep = ',' if '.csv' in file_path else '\t'
    for chunk in pd.read_csv(file_path, usecols=['sequence'], sep=sep, chunksize=batch_size):
        sequences = chunk['sequence'].tolist()
        tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in
                               sequences]
        tokenized_batch = np.vstack(tokenized_sequences)
        queue.put(tokenized_batch)
    queue.put(None)


