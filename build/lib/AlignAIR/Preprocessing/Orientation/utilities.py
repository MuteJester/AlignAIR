complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C','N':'N'}
def reverse_sequence(seq):
    return seq[::-1]
def complement_sequence(seq):
    return ''.join([complement[base] for base in seq])

def reverse_complement_sequence(seq):
    return complement_sequence(reverse_sequence(seq))


def single_fix_orientation(seq,orientation):
    if orientation == 'Normal':
        return seq
    elif orientation == 'Reversed':
        return reverse_sequence(seq)
    elif orientation == 'Complement':
        return complement_sequence(seq)
    elif orientation == 'Reverse Complement':
        return reverse_complement_sequence(seq)
    else:
        raise KeyError('Unrecognized Orientation Label')


def fix_orientation(pipeline,sequences):
    orts = pipeline.predict(sequences)

    fixed_sequences = []
    for sequence,ort in zip(sequences,orts):
        fixed_sequence = single_fix_orientation(sequence,ort)
        fixed_sequences.append(fixed_sequence)

    return fixed_sequences


