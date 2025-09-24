import numpy as np
import pandas as pd
from Bio import SeqIO

from AlignAIR.Preprocessing.Orientation import fix_orientation
from AlignAIR.Utilities.sequence_processing import encode_and_equal_pad_sequence




def sequence_tokenizer_worker(file_path, queue, max_seq_length, tokenizer_dictionary,logger,orientation_pipeline,candidate_extractor, batch_size=256):
    sep = ',' if '.csv' in file_path else '\t'
    for chunk in pd.read_csv(file_path, usecols=['sequence'], sep=sep, chunksize=batch_size):
        sequences = chunk['sequence'].tolist()

        # validate sequence length and use candidate extractor in case a long sequence is observed
        sequences = [seq if len(seq) < max_seq_length else candidate_extractor.transform_holt(seq)[0] for seq in sequences]


        # Fix Orientation
        if orientation_pipeline is not None:
            sequences = fix_orientation(orientation_pipeline,sequences)


        tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in
                               sequences]
        tokenized_batch = np.vstack(tokenized_sequences).astype(np.int32)
        queue.put((tokenized_batch,sequences))

    queue.put(None)


def sequence_tokenizer_worker_fasta(file_path, queue, max_seq_length, tokenizer_dictionary,logger,orientation_pipeline,candidate_extractor, batch_size=256):

    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))

        if len(sequences) == batch_size:
            # Fix Orientation
            # validate sequence length and use candidate extractor in case a long sequence is observed
            sequences = [seq if len(seq) < max_seq_length else candidate_extractor.transform_holt(seq)[0] for seq in sequences]
            if orientation_pipeline is not None:
                sequences = fix_orientation(orientation_pipeline, sequences)

            tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in sequences]
            tokenized_batch = np.vstack(tokenized_sequences).astype(np.int32)
            queue.put((tokenized_batch,sequences))
            sequences = []

    # Process any remaining sequences
    if sequences:
        # Fix Orientation
        # validate sequence length and use candidate extractor in case a long sequence is observed
        sequences = [seq if len(seq) < max_seq_length else candidate_extractor.transform_holt(seq)[0] for seq in sequences]

        if orientation_pipeline is not None:
            sequences = fix_orientation(orientation_pipeline, sequences)

        tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in sequences]
        tokenized_batch = np.vstack(tokenized_sequences).astype(np.int32)
        queue.put((tokenized_batch,sequences))

    queue.put(None)


READER_WORKER_TYPES = {
    'csv':sequence_tokenizer_worker,
    'tsv':sequence_tokenizer_worker,
    'fasta':sequence_tokenizer_worker_fasta
}