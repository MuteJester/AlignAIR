import pandas as pd
from Bio import SeqIO
from AlignAIR.Utilities.sequence_processing import tokenize_sequences


def count_rows(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        row_count = sum(1 for row in file)
    return row_count - 1

def count_sequences_in_fasta(file_path):
    """
    Counts the number of sequences in a FASTA file.

    Parameters:
    file_path (str): Path to the FASTA file.

    Returns:
    int: The number of sequences in the FASTA file.
    """
    sequence_count = sum(1 for _ in SeqIO.parse(file_path, "fasta"))
    return sequence_count

FILE_ROW_COUNTERS = {
    'csv':count_rows,
    'tsv':count_rows,
    'fasta':count_sequences_in_fasta
}

def process_and_tokenize_sequences(sequences, max_seq_length, tokenizer_dictionary):
    tokenized_matrix = tokenize_sequences(sequences, max_seq_length, tokenizer_dictionary, verbose=True)

    return tokenized_matrix
def fasta_sequence_generator(file_path, batch_size=256):
    """
    Generator that reads sequences from a FASTA file in batches.

    Parameters:
    file_path (str): Path to the FASTA file.
    batch_size (int): Number of sequences to yield per batch.

    Yields:
    list of str: A batch of sequences.
    """
    batch = []
    for record in SeqIO.parse(file_path, "fasta"):
        batch.append(str(record.seq))
        if len(batch) == batch_size:
            yield batch
            batch = []

    # Yield any remaining sequences
    if batch:
        yield batch

def tabular_sequence_generator(file_path, batch_size=256):
    sep = ',' if '.csv' in file_path else '\t'
    for chunk in pd.read_csv(file_path, usecols=['sequence'], sep=sep, chunksize=batch_size):
        yield chunk['sequence'].tolist()


def read_sequences_from_table(file_path):
    sep = ',' if '.csv' in file_path else '\t'
    return pd.read_csv(file_path, usecols=['sequence'], sep=sep)

FILE_SEQUENCE_GENERATOR = {
    'tsv': tabular_sequence_generator,
    'csv':tabular_sequence_generator,
    'fasta':fasta_sequence_generator
}