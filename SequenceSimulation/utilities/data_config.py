import csv
from collections import defaultdict
from typing import Optional, Dict

from SequenceSimulation.locus import LocusType, GenotypeLocus, HaplotypeLocus
from SequenceSimulation.utilities import create_family_use_dict, create_trimming_dict, create_NP_length_dict, \
    create_first_base_dict, create_NP_position_transition_dict, create_mut_rate_per_seq_dict, create_kmer_base_dict, \
    TrimMode, NP
from SequenceSimulation.utilities.data_utilities import create_allele_dict


class DataConfig:
    def __init__(self, data_folder: Optional[str] = None, mutate: bool = False, locus_type=LocusType.GENOTYPE,
                 haplotype_ratios=None,trim_modes=None,np_usage = NP.DEFAULT,gapped=False):
        self.haplotype_ratios = haplotype_ratios
        self.locus_type = locus_type
        self.data_folder = data_folder
        self.mutate = mutate

        # Config Variables
        self.family_use_dict = {}
        self.gene_use_dict = {}
        self.trim_dicts = {}
        self.NP_transitions = {}
        self.NP_first_bases = {}
        self.NP_lengths = {}
        self.mut_rate_per_seq = {}
        self.kmer_dicts = {}
        self.v_alleles = None
        self.d_alleles = None
        self.j_alleles = None
        self._load_data()

        # Settings
        self.trim_modes = {'V': TrimMode.DEFAULT, 'D': TrimMode.DEFAULT, 'J': TrimMode.DEFAULT} if trim_modes is None\
            else trim_modes
        self.np_usage = np_usage
        self.gapped = gapped

    def _load_gene_usage(self):
        # Family Usage
        v_usage = create_family_use_dict(
            f"{self.data_folder}/IGHV_usage.csv")
        d_usage = create_family_use_dict(
            f"{self.data_folder}/IGHD_usage.csv")
        j_usage = create_family_use_dict(
            f"{self.data_folder}/IGHJ_usage.csv")

        self.family_use_dict = {"V": v_usage, "D": d_usage, "J": j_usage}
        # Gene Usage
        v_gene_usage = create_family_use_dict(
            f"{self.data_folder}/IGHV_usage_gene.csv")

        d_gene_usage = create_family_use_dict(
            f"{self.data_folder}/IGHD_usage_gene.csv")

        j_gene_usage = create_family_use_dict(
            f"{self.data_folder}/IGHJ_usage_gene.csv")

        self.gene_use_dict = {"V": v_gene_usage, "D": d_gene_usage, "J": j_gene_usage}

    def _load_trimming_proportions(self):

        v_trim = create_trimming_dict(
            f"{self.data_folder}/V_family_trimming_proportions.csv")
        d_3_trim = create_trimming_dict(
            f"{self.data_folder}/D_3_family_trimming_proportions.csv")
        d_5_trim = create_trimming_dict(
            f"{self.data_folder}/D_5_family_trimming_proportions.csv")
        j_trim = create_trimming_dict(
            f"{self.data_folder}/J_family_trimming_proportions.csv")

        self.trim_dicts = {"V_3": v_trim, "D_5": d_5_trim, "D_3": d_3_trim, "J_5": j_trim}

    def _load_np_lengths(self):
        NP1_lengths = create_NP_length_dict(
            f"{self.data_folder}/np1_lengths_proportions.csv")
        NP2_lengths = create_NP_length_dict(
            f"{self.data_folder}/np2_lengths_proportions.csv")

        self.NP_lengths = {"NP1": NP1_lengths, "NP2": NP2_lengths}

    def _load_np_first_base_use(self):
        NP1_first_base_use = create_first_base_dict(
            f"{self.data_folder}/np1_first_base_probs.csv"
        )
        NP2_first_base_use = create_first_base_dict(
            f"{self.data_folder}/np2_first_base_probs.csv"
        )
        self.NP_first_bases = {"NP1": NP1_first_base_use, "NP2": NP2_first_base_use}

    def _load_np_transition_probabilities(self):
        NP1_transitions, NP2_transitions = None, None
        if self.mutate == False:
            NP1_transitions = create_NP_position_transition_dict(
                f"{self.data_folder}/np1_transition_probs_per_position_igdm.csv")

            NP2_transitions = create_NP_position_transition_dict(
                f"{self.data_folder}/np2_transition_probs_per_position_igdm.csv")

        if self.mutate == True:
            NP1_transitions = create_NP_position_transition_dict(
                f"{self.data_folder}/np1_transition_probs_per_position_igag.csv")

            NP2_transitions = create_NP_position_transition_dict(
                f"{self.data_folder}/np2_transition_probs_per_position_igag.csv")

        self.NP_transitions = {"NP1": NP1_transitions, "NP2": NP2_transitions}

    def _load_mut_rate_per_seq(self):
        mut_rate_per_seq = create_mut_rate_per_seq_dict(f"{self.data_folder}/mut_freq_per_seq_per_family.csv")
        self.mut_rate_per_seq = mut_rate_per_seq

    def _load_per_base_mutations(self):
        cdr1_kmers = create_kmer_base_dict(
            f"{self.data_folder}/cdr1_kmer_base_usage.csv")
        cdr2_kmers = create_kmer_base_dict(
            f"{self.data_folder}/cdr2_kmer_base_usage.csv")
        cdr3_kmers = create_kmer_base_dict(
            f"{self.data_folder}/cdr3_kmer_base_usage.csv")
        cdr_kmers = create_kmer_base_dict(
            f"{self.data_folder}/cdr_kmer_base_usage.csv")
        fwr1_kmers = create_kmer_base_dict(
            f"{self.data_folder}/fwr1_kmer_base_usage.csv")
        fwr2_kmers = create_kmer_base_dict(
            f"{self.data_folder}/fwr2_kmer_base_usage.csv")
        fwr3_kmers = create_kmer_base_dict(
            f"{self.data_folder}/fwr3_kmer_base_usage.csv")
        fwr4_kmers = create_kmer_base_dict(
            f"{self.data_folder}/fwr4_kmer_base_usage.csv")
        fwr_kmers = create_kmer_base_dict(
            f"{self.data_folder}/fwr_kmer_base_usage.csv")

        self.kmer_dicts = {"fwr1": fwr1_kmers, "fwr2": fwr2_kmers, "fwr3": fwr3_kmers, "fwr4": fwr4_kmers,
                           "cdr1": cdr1_kmers, "cdr2": cdr2_kmers, "cdr3": cdr3_kmers, "cdr": cdr_kmers,
                           "fwr": fwr_kmers}

    def _load_alleles(self):
        self.v_alleles = create_allele_dict(f"{self.data_folder}/imgt_human_IGHV.fasta")
        self.d_alleles = create_allele_dict(f"{self.data_folder}/imgt_human_IGHD.fasta")
        self.j_alleles = create_allele_dict(f"{self.data_folder}/imgt_human_IGHJ.fasta")

    def _get_allele_dict(self):
        return {'V': self.v_alleles, 'D': self.d_alleles, 'J': self.j_alleles}

    def _load_locus(self):
        if self.locus_type == LocusType.GENOTYPE:
            self.locus = GenotypeLocus(self._get_allele_dict())
        elif self.locus_type == LocusType.HAPLOTYPE:
            assert self.haplotype_ratios is not None
            self.locus = HaplotypeLocus(self._get_allele_dict(), het_list=self.haplotype_ratios)
        else:
            raise ValueError("Unknown Locus Type!")

    def _load_data(self):
        assert self.data_folder is not None
        # GENE USAGE
        self._load_gene_usage()

        # TRIMMING PROPORTIONS
        self._load_trimming_proportions()

        # N REGIONS LENGTHS
        self._load_np_lengths()
        # N REGIONS  FIRST BASE USAGE
        self._load_np_first_base_use()
        # N REGIONS MARKOV TRANSITION MATRICES
        self._load_np_transition_probabilities()

        # LOAD PER FAMILY MUTATION RATE PROPORTIONS
        self._load_mut_rate_per_seq()

        # PER BASE MUTATION
        self._load_per_base_mutations()

        # LOAD ALLELES
        self._load_alleles()

        # GENERATE LOCUS
        self._load_locus()
