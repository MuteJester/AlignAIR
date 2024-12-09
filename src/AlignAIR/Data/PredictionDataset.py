import numpy as np
import pandas as pd
from Bio import SeqIO


class PredictionDataset:

    def __init__(self, max_sequence_length):
        """
        This class is used to create a dataset for the prediction process of the AlignAIR models.
        It only loads in sequence and encodes them to the supported format for the model without preprocessing
        the variables needed for training.

        Note that this dataset object does not apply the various pre-processing steps that are required for proper
        alignment, for the full pipeline please refer to the prediction script in the AlignAIR module.

        Args:
            max_sequence_length: The maximum length of the sequence used to train the AlignAIR model.
        """
        self.max_sequence_length = max_sequence_length
        self.max_seq_length = max_sequence_length
        self.tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }

    def encode_and_equal_pad_sequence(self, sequence, return_padding=False):
        """Encodes a sequence of nucleotides and pads it to the specified maximum length, equally from both sides.

        Args:
            sequence: A sequence of nucleotides.

        Returns:
            A padded sequence, and the start and end indices of the unpadded sequence.
        """
        encoded_sequence = np.array([self.tokenizer_dictionary[i] for i in sequence])
        padding_length = self.max_seq_length - len(encoded_sequence)
        iseven = padding_length % 2 == 0
        pad_size = padding_length // 2
        if iseven:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size), 'constant', constant_values=(0, 0))
        else:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size + 1), 'constant', constant_values=(0, 0))

        if return_padding:
            return {'tokenized_sequence': encoded_sequence, 'padding': pad_size}
        else:
            return {'tokenized_sequence': encoded_sequence}

    def process_sequences(self, input_data):
        """
        Processes the input data, which can be a list of sequences or a file path.

        Args:
            input_data: A list of sequences or a file path to a CSV, TSV, or FASTA file.

        Returns:
            A list of encoded and padded sequences.
        """
        if isinstance(input_data, list):
            sequences = input_data
        elif isinstance(input_data, str):
            if input_data.endswith('.csv'):
                sequences = self._read_csv(input_data)
            elif input_data.endswith('.tsv'):
                sequences = self._read_tsv(input_data)
            elif input_data.endswith('.fasta'):
                sequences = self._read_fasta(input_data)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV, TSV, or FASTA file.")
        else:
            raise ValueError("Input data must be a list of sequences or a file path.")

        return [self.encode_and_equal_pad_sequence(seq) for seq in sequences]

    def _read_csv(self, file_path):
        """
        Reads a CSV file and extracts the sequences from the 'sequence' column.

        Args:
            file_path: Path to the CSV file.

        Returns:
            A list of sequences.
        """
        df = pd.read_csv(file_path)
        if 'sequence' not in df.columns:
            raise ValueError("CSV file must contain a 'sequence' column.")
        return df['sequence'].tolist()

    def _read_tsv(self, file_path):
        """
        Reads a TSV file and extracts the sequences from the 'sequence' column.

        Args:
            file_path: Path to the TSV file.

        Returns:
            A list of sequences.
        """
        df = pd.read_table(file_path)
        if 'sequence' not in df.columns:
            raise ValueError("TSV file must contain a 'sequence' column.")
        return df['sequence'].tolist()

    def _read_fasta(self, file_path):
        """
        Reads a FASTA file and extracts the sequences.

        Args:
            file_path: Path to the FASTA file.

        Returns:
            A list of sequences.
        """
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))
        return sequences
