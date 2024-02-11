import csv
import re
from collections import defaultdict

from SequenceSimulation.alleles.allele import VAllele, JAllele, DAllele
from SequenceSimulation.utilities import parse_fasta


def create_family_use_dict(usage_csv):
    """Creates a dictionary for V, D or J family usage.

    Args:
        usage_csv (file): A CSV file consisting of gene family and proportion
            of sequences to use alleles from that gene family.

    Raises:
        ValueError: Warns of unexpected file format - checks line length.

    Returns:
        family_dict (dict): Dictionary of gene families and the proportion of
            sequences to use their alleles. In the format {gene family: proportion}
    """
    with open(usage_csv, "r") as f:
        family_dict = {}
        csvlines = csv.reader(f, delimiter=",")
        next(csvlines)  # skip header
        for line in csvlines:
            if not len(line) == 2:
                raise ValueError(
                    f"Invalid csv format. Expected line length 2. Line {csvlines.line_num} has length {len(line)}.")
            else:
                family, prop = line
                family_dict[family] = float(prop)
        return dict(family_dict)


def create_trimming_dict(trim_csv):
    """Creates a dictionary of V,D or J trimming proportions per gene family.

    Args:
        trim_csv (file): A CSV file consisting of IGH gene family (e.g. IGHV1),
            number of trimmed nucleotides, and proportion of repertoire trimmed in
            such a way.

    Raises:
        ValueError: Warns of unexpected file format - checks line length.

    Returns:
        trim_dict (dict): A nested dictionary containing trimming distributions for each gene
            family. In the format {Gene family: {Trim length : proportion of sequences}}.
    """
    with open(trim_csv, "r") as f:

        trim_dict = defaultdict(lambda: defaultdict(float))

        csvlines = csv.reader(f, delimiter=",")
        next(csvlines)  # skip header
        for line in csvlines:
            if not len(line) == 3:
                raise ValueError(
                    f"Invalid csv format. Expected line length 3. Line {csvlines.line_num} has length {len(line)}.")
            else:
                family, trim, prop = line  # ! Assumes column order in file
                trim_dict[family][int(trim)] = float(prop)
        return dict(trim_dict)


def create_NP_length_dict(lengths_csv):
    """Creates a dictionary that contains distributions of NP region lengths.

    Args:
        lengths_csv (file): A CSV file consisting of NP region length and the
            proportion of sequences to use this length.

    Raises:
        ValueError: Warns of unexpected file format - checks line length.

    Returns:
        lengths_dict (dict): Dictionary of possible NP region lengths and the proportion of
            sequences to use them. In the format {NP region length: proportion}
    """
    with open(lengths_csv, "r") as f:
        lengths_dict = {}
        csvlines = csv.reader(f, delimiter=",")
        next(csvlines)  # skip header
        for line in csvlines:
            if not len(line) == 2:
                raise ValueError(
                    f"Invalid csv format. Expected line length 2. Line {csvlines.line_num} has length {len(line)}.")
            else:
                length, prop = line
                lengths_dict[int(float(length))] = float(prop)
        return dict(lengths_dict)


def create_first_base_dict(first_base_csv):
    """Creates a dictionary for choosing the first base of an NP region.

    Args:
        first_base_csv (file): A CSV file consisting of nucleotide base
            (A,C,G,T) and the probability of NP regions using this nucleotide
            as the first base.

    Raises:
        ValueError: Warns of unexpected file format - checks line length.

    Returns:
        first_base_dict (dict): Dictionary of nucleotide bases and the
            probability of NP regions using this nucleotide as the first base.
            In the format {Nucleotide base: probability, ...}
    """

    with open(first_base_csv, "r") as f:
        first_base_dict = {}
        csvlines = csv.reader(f, delimiter=",")
        next(csvlines)  # skip header
        for line in csvlines:
            if not len(line) == 2:
                raise ValueError(
                    f"Invalid csv format. Expected line length 2. Line {csvlines.line_num} has length {len(line)}.")
            else:
                base, prop = line
                first_base_dict[base] = float(prop)
        return dict(first_base_dict)


def create_NP_position_transition_dict(transitions_csv):
    """Creates a dictionary representing a transition matrix for NP region.

    Creates a dictionary representing a transition matrix with the
    probabilities of transitioning from one nucleotide base (A,C,G,T) to another
    within the NP region based on the current position in the sequence.

    Args:
        transitions_csv (file): A CSV file consisting of nucleotide base
            (A,C,G,T) and the probability of transitioning to each other base for
            each position in the NP region

    Raises:
        ValueError: Warns of unexpected file format - checks line length.

    Returns:
        NP_len_transition_dict (dict): Dictionary of nucleotide bases and
            the probability of transitioning to each other base at each position
            in NP region. In the format
            {Position: {Nucleotide base:
                    {Nucleotide base: probability, ...}, ...}, ...}
    """

    with open(transitions_csv, "r") as f:

        NP_len_transition_dict = defaultdict(lambda: defaultdict(dict))

        csvlines = csv.reader(f, delimiter=",")
        next(csvlines)  # skip header
        for line in csvlines:
            if not len(line) == 6:
                raise ValueError(
                    f"Invalid csv format. Expected line length 6. Line {csvlines.line_num} has length {len(line)}.")
            else:
                position, base, A, C, G, T = line
                NP_len_transition_dict[int(float(position))][base] = {
                    "A": float(A),
                    "C": float(C),
                    "G": float(G),
                    "T": float(T),
                }
        return dict(NP_len_transition_dict)


def create_mut_rate_per_seq_dict(mute_per_seq_csv):
    mut_rate_per_seq = defaultdict(dict)
    with open(mute_per_seq_csv, newline='') as f:
        csvlines = csv.reader(f, delimiter=",")
        next(csvlines)  # skip header
        for line in csvlines:
            if not len(line) == 3:
                raise ValueError(
                    f"Invalid csv format. Expected line length 3. Line {csvlines.line_num} has length {len(line)}.")
            else:
                family, mut_rate, prop = line
                mut_rate_per_seq[family][float(mut_rate)] = float(prop)
    return mut_rate_per_seq


def create_kmer_base_dict(kmer_csv):
    """
    Creates a dictionary of kmer mutation probabilities.

    Args:
        kmer_csv (file): A CSV file consisting of unique kmer and the
            proportion of sequences using each nucleotide base (A, T, C, G)
            at the centre position of this kmer.

    Returns:
        kmer_dict (dict): Nested dictionary containing nucleotide distributions for the
            centre position of each kmer. In the format
            {kmer: {A : Proportion, T : Proportion,
            C : Proportion, G : Proportion }}
    """

    with open(kmer_csv, "r") as f:
        kmer_dict = defaultdict(lambda: defaultdict(dict))

        csvlines = csv.reader(f, delimiter=",")
        next(csvlines)  # skip header
        for line in csvlines:
            if not len(line) == 5:
                raise ValueError(
                    f"Invalid csv format. Expected line length 5. Line {csvlines.line_num} has length {len(line)}.")
            else:
                kmer, A, C, G, T = line
                kmer_dict[kmer.lower()] = {
                    "A": float(A),
                    "C": float(C),
                    "G": float(G),
                    "T": float(T),
                }
        return dict(kmer_dict)

def create_allele_dict(fasta):
    """Creates an allele dictionary from an IMGT formatted reference FASTA.

    Args:
        fasta (file): A FASTA file consisting of reference Ig V, D or J alleles,
            with IMGT-gapped sequences. Header is expected to follow IMGT format.

    Returns:
        allele_dict (dict): Allele dictionary with V, D or J gene as keys and
            lists of the corresponding alleles, in the form of Allele class
            instances, as values.
            Only returns full length alleles denoted as functional or open reading
            frame by the IMGT (marked by "F" or "ORF" in the sequence header).
            {Gene : [Allele, Allele ...]}
    """

    # TODO - add exception handling here to check expected allele format etc
    # TODO - remove duplicate sequence from reference example: IGHV3-30-3*03 has identical
    #  sequence to IGHV3-30*04 and IGHV5-10-1*02 has anchor Cys that makes it out of frame

    allele_dict = defaultdict(list)
    allele_class_map = {'V':VAllele,'D':DAllele,'J':JAllele}
    with open(fasta) as f:
        for header, seq in parse_fasta(f):
            
            header = header.replace(">","") # clean the fasta sequence tag
            seq = seq.lower() # make sure the sequences is in lower cases
            
            allele = header.split("|") # leave the IMGT option, but allow OGRDB reference
            if len(allele) == 1:
                allele = allele[0]
            else:
                allele = allele[1]
                
            segment = None

            if "-" in allele:
                family = allele.split("-")[0]
            else:
                family = allele.split("*")[0]
            if "D" in family:
                seq = seq.replace(".", "").lower()
                segment = 'D'
            if "V" in family:
                cys = seq[309:312]
                if cys != "tgt" and cys != "tgc":  # if allele doesn't have expected V anchor then don't use it
                    continue
                segment = 'V'
            if "J" in family:  # if allele doesn't have expected J anchor then don't use it
                motif = re.compile('(ttt|ttc|tgg)(ggt|ggc|gga|ggg)[a-z]{3}(ggt|ggc|gga|ggg)')
                match = motif.search(seq)
                if match is None:
                    continue
                segment = 'J'

            gene = allele.split("*")[0]
            
            coding = header.split("|") # extract the coding tag from IMGT references, OGRDB only include functional sequences 
            if len(coding) == 1:
                coding = "F"
            else:
                coding = coding[3]  
                
            ungapped_length = len(seq.replace(".","")) # get the length of the sequence
            
            if "partial" not in header and (coding == "F" or coding == "ORF"):
                if segment is not None:
                    allele_dict[gene].append(allele_class_map[segment](allele, seq, ungapped_length))

    return dict(allele_dict)