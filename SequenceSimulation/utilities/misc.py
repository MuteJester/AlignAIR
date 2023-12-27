import random


def weighted_choice(choices):
    return random.choices(population=list(choices.keys()), weights=choices.values(), k=1)[0]


def weighted_choice_zero_break(choices):
    """
    return zero if choices is empty
    :param choices:
    :return:
    """
    if len(choices) < 1:
        return 0
    else:
        return weighted_choice(choices)


def translate(seq):
    """Translates a nucleotide sequence to an amino acid sequence.

    Args:
        seq (str): Nucleotide sequence to be translated.

    Returns:
        protein (str): Amino acid sequence.
    """

    table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
        'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W'
    }
    protein = ""
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3].upper()
        if len(codon) < 3:
            protein += '.'
        elif "." in codon:
            protein += '_'
        elif "-" in codon:
            protein += 'X'
        elif "N" in codon:
            protein += 'X'
        else:
            protein += table[codon]
    return protein
