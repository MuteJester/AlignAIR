import numpy as np


class HeavyChainInputPreProcessor:
    def __init__(self, heavy_chain_dataconfig, max_sequence_length=576):
        """
        Standalone PyTorch dataset class for heavy chain sequences.
        """
        self.dataconfig = heavy_chain_dataconfig
        self.max_sequence_length = max_sequence_length
        self.column_types = {
            'sequence': str, 'v_sequence_start': np.float32, 'v_sequence_end': np.float32,
            'd_sequence_start': np.float32, 'd_sequence_end': np.float32, 'j_sequence_start': np.float32,
            'j_sequence_end': np.float32, 'v_call': str, 'd_call': str, 'j_call': str,
            'mutation_rate': np.float32, 'indels': eval, 'productive': lambda x: np.float32(bool(x))
        }
        self.required_data_columns = list(self.column_types.keys())
        self.tokenizer_dictionary = {
            "A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0  # Pad token
        }

        # Initialize dictionaries and one-hot encodings
        self._derive_call_dictionaries()
        self._derive_call_one_hot_representation()

    def _derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        j_alleles = sorted(list(self.j_dict))
        # Add Short D Label as Last Label
        d_alleles = d_alleles + ['Short-D']

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe}
        }

    def _derive_call_dictionaries(self):
        self.v_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.v_alleles for j in
                       self.dataconfig.v_alleles[i]}
        self.d_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.d_alleles for j in
                       self.dataconfig.d_alleles[i]}
        self.j_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.j_alleles for j in
                       self.dataconfig.j_alleles[i]}

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def _encode_and_equal_pad_sequence(self, sequence):
        encoded_sequence = [self.tokenizer_dictionary.get(i, 0) for i in sequence]
        padding_length = self.max_sequence_length - len(encoded_sequence)
        pad_size = padding_length // 2
        iseven = (padding_length % 2) == 0
        if iseven:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size), constant_values=0)
        else:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size + 1), constant_values=0)
        return encoded_sequence, pad_size

    def _encode_and_pad_sequences(self, sequences):
        padded_arrays = []
        paddings = []
        for seq in sequences:
            padded, padding = self._encode_and_equal_pad_sequence(seq)
            padded_arrays.append(padded)
            paddings.append(padding)
        return np.array(padded_arrays), np.array(paddings)

    def one_hot_encode_allele(self, gene, alleles):
        allele_count = self.properties_map[gene]['allele_count']
        allele_call_ohe = self.properties_map[gene]['allele_call_ohe']
        ohe = np.zeros(allele_count).astype(np.float32)
        for allele in alleles:
            if allele in allele_call_ohe:
                ohe[allele_call_ohe[allele]] = 1
        return ohe

    def cast_types(self, sample):
        processed_sample = {}
        for key, value in sample.items():
            if key in self.required_data_columns:
                processed_sample[key] = self.column_types[key](value)

        return processed_sample

    def __call__(self, sample):

        casted_sample = self.cast_types(sample)
        encoded_sequences, paddings = self._encode_and_equal_pad_sequence(casted_sample['sequence'])

        for _gene in ['v_sequence', 'd_sequence', 'j_sequence']:
            for _position in ['start', 'end']:
                casted_sample[_gene + '_' + _position] += paddings
        x = encoded_sequences

        v_alleles = casted_sample['v_call'].split(',')
        d_alleles = casted_sample['d_call'].split(',')
        j_alleles = casted_sample['j_call'].split(',')

        y = {
            "v_start": casted_sample['v_sequence_start'],
            "v_end": casted_sample['v_sequence_end'],
            "d_start": casted_sample['d_sequence_start'],
            "d_end": casted_sample['d_sequence_end'],
            "j_start": casted_sample['j_sequence_start'],
            "j_end": casted_sample['j_sequence_end'],
            "v_allele": self.one_hot_encode_allele("V", v_alleles),
            "d_allele": self.one_hot_encode_allele("D", d_alleles),
            "j_allele": self.one_hot_encode_allele("J", j_alleles),
            'mutation_rate': casted_sample['mutation_rate'],
            'indel_count': len(casted_sample['indels']),
            'productive': casted_sample['productive']

        }

        return {'x': x, 'y': y}


class SequenceTokenizer:
    def __init__(self, heavy_chain_dataconfig, max_sequence_length=576,return_original_sequence=False):
        self.dataconfig = heavy_chain_dataconfig
        self.max_sequence_length = max_sequence_length
        self.return_original_sequence = return_original_sequence
        self.tokenizer_dictionary = {
            "A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0  # Pad token
        }

    def _encode_and_equal_pad_sequence(self, sequence):
        encoded_sequence = [self.tokenizer_dictionary.get(i, 0) for i in sequence]
        padding_length = self.max_sequence_length - len(encoded_sequence)
        pad_size = padding_length // 2
        iseven = (padding_length % 2) == 0
        if iseven:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size), constant_values=0)
        else:
            encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size + 1), constant_values=0)

        return encoded_sequence

    def __call__(self, sample):
        encoded_sequence = self._encode_and_equal_pad_sequence(sample['sequence'])
        if self.return_original_sequence:
            return {'x': encoded_sequence,'x_original':sample['sequence']}
        else:
            return {'x': encoded_sequence}


