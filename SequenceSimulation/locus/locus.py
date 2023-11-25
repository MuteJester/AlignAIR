import random
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto

class LocusType(Enum):
    HAPLOTYPE = auto()
    GENOTYPE = auto()


def random_sequence(locus, gene_use_dict, family_use_dict, segments=["V", "D", "J"], flat=False):
    """Creates a random sequence.

    Initialises a Sequence object from randomly chosen alleles from a previously
    created locus. The gene to pick alleles from is randomly chose based on the
    desired gene usage distribution.

    Args:
        locus (list): List of two dictionaries. Each is a dictionary containing
            the gene segment as keys and the chosen alleles as values. Format is
            [{Segment : [Allele, Allele ...], ...}, {Segment : [Allele, Allele ...], ...}]
        gene_use_dict (_type_): Nested dictionary of gene segment and
            genes and the proportion of sequences to use their alleles.
            In the format {segment: {gene: proportion, ...}, ...}
        family_use_dict (_type_): Nested dictionary of gene segment and
            gene families and the proportion of sequences to use their alleles.
            In the format {segment: {gene family: proportion, ...}, ...}
        segments (list, optional): Gene segments to include in sequence.
            Defaults to ["V", "D", "J"].
        flat (optional): gene, family or False. Gene or family specify that
            sequences should use all genes or gene families evenly. If false,
            usage follows experimental distributions. Defaults to False.

    Returns:
        out_sequence (Sequence): Randomly generated Sequence class object.
    """

    alleles = []
    chromosome = random.choice(locus)
    for segment in segments:
        if flat == False:  # if family usage is to be biased
            counter = 1
            while counter > 0:
                gene_dict = gene_use_dict[segment]
                gene = weighted_choice(gene_dict.items())
                gene_alleles = [
                    x for x in chromosome[segment] if x.gene == gene]
                if len(gene_alleles) > 0:
                    allele = random.choice(gene_alleles)
                    counter = - 1
                    alleles.append(allele)
        elif flat == "gene":
            counter = 1
            while counter > 0:
                gene_dict = gene_use_dict[segment]
                gene = random.choice(list(gene_dict.keys()))
                gene_alleles = [
                    x for x in chromosome[segment] if x.gene == gene]
                if len(gene_alleles) > 0:
                    allele = random.choice(gene_alleles)
                    counter = - 1
                    alleles.append(allele)
        elif flat == "family":  # if family usage is to be even
            family_dict = family_use_dict[segment]
            families = list(family_dict.keys())
            family = random.choice(families)
            allele = random.choice(
                [x for x in chromosome[segment] if x.family == family])
            alleles.append(allele)
        elif flat == "allele":  # if allele usage is to be even
            allele = random.choice(chromosome[segment])
            alleles.append(allele)
    out_sequence = Sequence(alleles[0], alleles[1], alleles[2])
    return out_sequence



class LocusBase(ABC):
    def __init__(self, allele_dicts):
        self.allele_dicts = allele_dicts
        self.segments = list(allele_dicts)
        self.chromosome1 = defaultdict(list)
        self.chromosome2 = defaultdict(list)

    @abstractmethod
    def create_locus(self):
        pass  # This method must be implemented by subclasses.

    def get_locus(self):
        return [self.chromosome1, self.chromosome2]


class HaplotypeLocus(LocusBase):
    def __init__(self, allele_dicts, het_list=None):
        super().__init__(allele_dicts)
        if het_list is None:
            het_list = [1, 1, 1]
        self.het_list = het_list
        self.create_locus()

    def _create_locus_segment(self, allele_dict, hetero_prop):
        """Creates a two chromosome gene locus containing V, D and J alleles.

                Args:
                    segments (str): Which segments to include, expect V, D and J.
                        allele_dicts (dict): Dictionary of dictionaries, in the format of
                        {Segment : {Gene : [Allele, Allele ...]}}
                    hetero_props (list): List of integer proportion of positions to be
                        heterozygous for each position

                Returns:
                    locus (list): List of two dictionaries. Each is a dictionary containing the gene
                        segment as keys and the chosen alleles as values. Format is
                        {Segment : [Allele, Allele ...], ...}
                """
        chromosome1_segment, chromosome2_segment = [], []

        het = int(round(hetero_prop * len(allele_dict)))

        for i, alleles in enumerate(sorted(allele_dict.values(), key=lambda x: random.random())):
            allele = random.choice(alleles)
            chromosome1_segment.append(allele)
            if len(alleles) > 1 and het > 0:
                alleles.remove(allele)
                het_allele = random.choice(alleles)
                chromosome2_segment.append(het_allele)
                het -= 1
            else:
                chromosome2_segment.append(allele)
        return chromosome1_segment, chromosome2_segment

    def create_locus(self):
        for segment, hetero_prop in zip(self.segments, self.het_list):
            self.chromosome1[segment], self.chromosome2[segment] = self._create_locus_segment(
                self.allele_dicts[segment], hetero_prop
            )


class GenotypeLocus(LocusBase):
    def __init__(self, allele_dicts):
        super().__init__(allele_dicts)
        self.create_locus()

    def create_locus(self):
        for segment in self.segments:
            allele_dict = self.allele_dicts[segment]
            for gene in allele_dict.values():
                for allele in gene:
                    self.chromosome1[segment].append(allele)
                    self.chromosome2[segment].append(allele)


