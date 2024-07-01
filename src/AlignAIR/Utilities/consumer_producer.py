import numpy as np
import pandas as pd
from Bio import SeqIO

from AlignAIR.Preprocessing.Orientation import fix_orientation
from AlignAIR.Utilities.sequence_processing import encode_and_equal_pad_sequence




def sequence_tokenizer_worker(file_path, queue, max_seq_length, tokenizer_dictionary,logger,orientation_pipeline, batch_size=256):
    sep = ',' if '.csv' in file_path else '\t'
    for chunk in pd.read_csv(file_path, usecols=['sequence'], sep=sep, chunksize=batch_size):
        sequences = chunk['sequence'].tolist()

        # Fix Orientation
        if orientation_pipeline is not None:
            sequences = fix_orientation(orientation_pipeline,sequences)

        tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in
                               sequences]
        tokenized_batch = np.vstack(tokenized_sequences)
        queue.put((tokenized_batch,sequences))
    queue.put(None)


def sequence_tokenizer_worker_fasta(file_path, queue, max_seq_length, tokenizer_dictionary,logger,orientation_pipeline, batch_size=256):

    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        if len(str(record.seq)) < max_seq_length:
            sequences.append(str(record.seq))
        else:
            logger.warning('Encountered a sequence that is longer than the maximum length of the model, skipping it...')
        if len(sequences) == batch_size:
            # Fix Orientation
            if orientation_pipeline is not None:
                sequences = fix_orientation(orientation_pipeline, sequences)

            tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in sequences]
            tokenized_batch = np.vstack(tokenized_sequences)
            queue.put((tokenized_batch,sequences))
            sequences = []

    # Process any remaining sequences
    if sequences:
        # Fix Orientation
        if orientation_pipeline is not None:
            sequences = fix_orientation(orientation_pipeline, sequences)

        tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in sequences]
        tokenized_batch = np.vstack(tokenized_sequences)
        queue.put((tokenized_batch,sequences))

    queue.put(None)


READER_WORKER_TYPES = {
    'csv':sequence_tokenizer_worker,
    'tsv':sequence_tokenizer_worker,
    'fasta':sequence_tokenizer_worker_fasta
}