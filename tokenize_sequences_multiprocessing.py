import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import pickle


def _process_and_dpad(sequence, max_seq_length, tokenizer_dictionary):
    trans_seq = [tokenizer_dictionary.get(i, 0) for i in sequence]  # Use .get() to handle unknown characters
    gap = max_seq_length - len(trans_seq)
    iseven = gap % 2 == 0
    whole_half_gap = gap // 2

    if iseven:
        trans_seq = [0] * whole_half_gap + trans_seq + ([0] * whole_half_gap)
    else:
        trans_seq = [0] * (whole_half_gap + 1) + trans_seq + ([0] * whole_half_gap)

    return trans_seq


def tokenize_chunk(chunk, max_seq_length, tokenizer_dictionary):
    return [(index, _process_and_dpad(sequence, max_seq_length, tokenizer_dictionary)) for index, sequence in chunk]


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


def process_csv_and_tokenize(csv_path,save_path, max_seq_length, tokenizer_dictionary):
    if '.csv' in csv_path:
        df = pd.read_csv(csv_path, usecols=['sequence'])
    elif '.tsv' in csv_path:
        df = pd.read_table(csv_path, usecols=['sequence'])

    tokenized_matrix = tokenize_sequences(df['sequence'], max_seq_length, tokenizer_dictionary, verbose=True)

    pickle_filename = save_path + csv_path.rsplit('.', 1)[0].split('/')[-1] + "_tokenized.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(tokenized_matrix, f)

    print(f"Tokenized sequences saved to {pickle_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize sequences from a CSV or TSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV or TSV file containing sequences.')
    parser.add_argument('max_size', type=int, help='Maximum size for the sequence padding.')
    parser.add_argument('save_path', type=str, help='Path to save tokenized result')
    args = parser.parse_args()

    tokenizer_dictionary =  {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }

    process_csv_and_tokenize(args.file_path,args.save_path, args.max_size, tokenizer_dictionary)
