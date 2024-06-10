import pandas as pd

from AlignAIR.Utilities.sequence_processing import tokenize_sequences


def count_rows(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        row_count = sum(1 for row in file)
    return row_count - 1

def process_and_tokenize_sequences(sequences, max_seq_length, tokenizer_dictionary):
    tokenized_matrix = tokenize_sequences(sequences, max_seq_length, tokenizer_dictionary, verbose=True)

    return tokenized_matrix

def tabular_sequence_generator(file_path, batch_size=256):
    sep = ',' if '.csv' in file_path else '\t'
    for chunk in pd.read_csv(file_path, usecols=['sequence'], sep=sep, chunksize=batch_size):
        yield chunk['sequence'].tolist()


def read_sequences_from_table(file_path):
    sep = ',' if '.csv' in file_path else '\t'
    return pd.read_csv(file_path, usecols=['sequence'], sep=sep)
