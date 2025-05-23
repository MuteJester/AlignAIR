from GenAIRR.dataconfig import DataConfig
from GenAIRR.utilities.data_utilities import create_allele_dict
from GenAIRR.utilities.asc_utilities import create_asc_germline_set
from GenAIRR.utilities import AlleleNComparer
import pickle


class RandomDataConfigGenerator:
    def __init__(self, convert_to_asc=True):
        self.convert_to_asc = convert_to_asc
        self.v_asc_table = None
        self.has_d = False
        # initialize an empty dataconfig object this class will populate
        self.dataconfig = DataConfig()
        # initialize auxiliary list for allele tracking
        self.alleles = ['V', 'J']

    def _get_reference_pointers(self):
        # aux list
        pointer_to_reference = {'V': self.dataconfig.v_alleles, 'J': self.dataconfig.j_alleles}
        if self.has_d:
            pointer_to_reference['D'] = self.dataconfig.d_alleles
        return pointer_to_reference

    def generate_decaying_probabilities(self, n, base=0.8):
        probabilities = {}
        total_sum = sum(base ** i for i in range(n + 1))
        for i in range(n + 1):
            probabilities[i] = (base ** i) / total_sum
        return probabilities

    def _load_alleles(self, v_alleles, j_alleles, d_alleles=None):
        self.dataconfig.v_alleles = v_alleles
        self.dataconfig.d_alleles = d_alleles
        self.dataconfig.j_alleles = j_alleles

    def _load_random_gene_usage(self):

        pointer_to_reference = self._get_reference_pointers()

        gene_use_dict = {'V': dict(), 'J': dict()}
        if self.has_d:
            gene_use_dict['D'] = dict()

        for allele in self.alleles:
            # get all alleles from reference per gene
            current_alleles = list(pointer_to_reference[allele])
            n = len(current_alleles)
            # uniformly attribute probability to each allele
            random_gene_usage = {i: (1 / n) for i in current_alleles}
            gene_use_dict[allele] = random_gene_usage

        self.dataconfig.gene_use_dict = gene_use_dict

    def _load_random_trimming_proportions(self, max_trim=50):
        pointer_to_reference = self._get_reference_pointers()
        trim_dicts = dict()
        for allele in self.alleles:
            # get only the families
            allele_families = sorted(list(set([i.split('-')[0] for i in pointer_to_reference[allele]])))
            # each family gets up to max_trim trimming amount options with decaying probabilities
            probabilities = self.generate_decaying_probabilities(max_trim)
            trim_dict = {i: probabilities for i in allele_families}

            trim_dicts[allele + '_5'] = trim_dict
            trim_dicts[allele + '_3'] = trim_dict

        self.dataconfig.trim_dicts = trim_dicts

    def _load_random_np_lengths(self, max_size=50):
        NP_lengths = dict()
        np_regions = ['NP1']
        if self.has_d:
            np_regions.append('NP2')

        for np_region in np_regions:
            probabilities = self.generate_decaying_probabilities(max_size)
            NP_lengths[np_region] = probabilities

        self.dataconfig.NP_lengths = NP_lengths

    def _load_random_np_first_base_use(self):
        NP_first_bases = dict()
        np_regions = ['NP1']
        if self.has_d:
            np_regions.append('NP2')

        for np_region in np_regions:
            base_probability = 1 / 4
            NP_first_bases[np_region] = {i: base_probability for i in ['A', 'T', 'C', 'G']}

        self.dataconfig.NP_first_bases = NP_first_bases

    def _load_random_np_transition_probabilities(self, max_size=50):
        NP_transitions = dict()
        nucleotides = ['A', 'T', 'C', 'G']
        np_regions = ['NP1']
        if self.has_d:
            np_regions.append('NP2')

        for np_region in np_regions:
            NP_transitions[np_region] = dict()
            for position in range(max_size):
                NP_transitions[np_region][position] = dict()
                for base_at_position in nucleotides:
                    # uniform transition probabilities
                    actions = {next_base: 1 / 4 for next_base in nucleotides}
                    NP_transitions[np_region][position][base_at_position] = actions

        self.dataconfig.NP_transitions = NP_transitions

    def _derive_3_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        trim_map = dict()
        for t_allele in t_dict.values():
            trim_map[t_allele.name] = dict()
            seq_length = len(t_allele.ungapped_seq)
            for trim_3 in range(seq_length + 1):
                # Trim from the right (3' end)
                trimmed = t_allele.ungapped_seq[:seq_length - trim_3] if trim_3 > 0 else t_allele.ungapped_seq
                trim_map[t_allele.name][seq_length - trim_3] = []
                for v_c_allele in t_dict.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[t_allele.name][seq_length - trim_3].append(v_c_allele.name)
        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_3_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_5_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}

        trim_map = dict()
        for t_allele in t_dict.values():
            trim_map[t_allele.name] = dict()
            for trim_5 in range(len(t_allele.ungapped_seq) + 1):
                trimmed = t_allele.ungapped_seq[trim_5:] if trim_5 > 0 else t_allele.ungapped_seq
                trim_map[t_allele.name][trim_5] = []
                for v_c_allele in t_dict.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[t_allele.name][trim_5].append(v_c_allele.name)
        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_5_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_5_and_3_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        t_list = [i for j in target_alleles for i in target_alleles[j]]
        trim_map = dict()
        for t_allele in t_list:
            trim_map[t_allele.name] = dict()
            for trim_5 in range(len(t_allele.ungapped_seq) + 1):
                for trim_3 in range(len(t_allele.ungapped_seq) - trim_5 + 1):
                    # Correctly handle the trimming for t_allele
                    trimmed = t_allele.ungapped_seq[trim_5:] if trim_5 > 0 else t_allele.ungapped_seq
                    trimmed = trimmed[:-trim_3] if trim_3 > 0 else trimmed

                    trim_map[t_allele.name][(trim_5, trim_3)] = []
                    for d_c_allele in t_list:
                        # Check if the trimmed sequence is a substring of the d_c_allele sequence
                        if trimmed in d_c_allele.ungapped_seq:
                            trim_map[t_allele.name][(trim_5, trim_3)].append(d_c_allele.name)

        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_5_3_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_n_ambiguity_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        comparer = AlleleNComparer()
        for v in t_dict:
            comparer.add_allele(v, t_dict[v].ungapped_seq.upper())

        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_N_AMBIGUITY_CORRECTION_GRAPH'] = comparer

    def make_dataconfig_from_existing_reference_files(self, v_reference_path, j_reference_path, d_reference_path=None):

        # update d flag
        self.has_d = d_reference_path is not None
        user_d_reference = None
        if self.has_d:
            # add D to aux list to calculate properties for D allele as well
            self.alleles.append('D')

        # 1. read fasta references
        if self.convert_to_asc:
            # ASC logic goes here to resulting variables should be of the following foramt:
            user_v_reference, v_asc_table = create_asc_germline_set(v_reference_path, segment="V")
            # save asc table so reverse transformation will be available to the user
            self.dataconfig.asc_tables['V'] = v_asc_table

            user_j_reference = create_allele_dict(j_reference_path)
            if self.has_d:
                user_d_reference = create_allele_dict(d_reference_path)
        else:

            user_v_reference = create_allele_dict(v_reference_path)
            if d_reference_path is not None:
                user_d_reference = create_allele_dict(d_reference_path)
            user_j_reference = create_allele_dict(j_reference_path)

        print('=' * 50)
        # 2. Fill in Data Config

        # LOAD ALLELES
        self._load_alleles(v_alleles=user_v_reference, d_alleles=user_d_reference, j_alleles=user_j_reference)
        print('Alleles Mounted to DataConfig!...')
        # RANDOM GENE USAGE
        self._load_random_gene_usage()
        print('Random Gene Usage Mounted to DataConfig!...')

        # TRIMMING PROPORTIONS
        self._load_random_trimming_proportions()
        print('Random Trimming Proportions Mounted to DataConfig!...')

        # N REGIONS LENGTHS
        self._load_random_np_lengths()
        print('Random NP Region Lengths Mounted to DataConfig!...')

        # N REGIONS  FIRST BASE USAGE
        self._load_random_np_first_base_use()
        print('Random NP Initial States Mounted to DataConfig!...')
        # N REGIONS MARKOV TRANSITION MATRICES
        self._load_random_np_transition_probabilities()
        print('Random NP Markov Chain Mounted to DataConfig!...')

        # 3. Fill in Data Config correction maps
        self._derive_n_ambiguity_map(self.dataconfig.v_alleles)
        print('V Ns Ambiguity Map Mounted to DataConfig!...')

        self._derive_3_prime_correction_map(self.dataconfig.v_alleles)
        print('V 3 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_5_prime_correction_map(self.dataconfig.v_alleles)
        print('V 5 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_3_prime_correction_map(self.dataconfig.j_alleles)
        print('J 3 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_5_prime_correction_map(self.dataconfig.j_alleles)
        print('J 5 Prime Ambiguity Map Mounted to DataConfig!...')
        if self.has_d:
            self._derive_5_and_3_prime_correction_map(self.dataconfig.d_alleles)
            print('D (5,3) Prime Ambiguity Map Mounted to DataConfig!...')

        print('=' * 50)

        return self.dataconfig


class CustomDataConfigGenerator:
    def __init__(self, convert_to_asc=True):
        self.convert_to_asc = convert_to_asc
        self.v_asc_table = None
        self.has_d = False
        # initialize an empty dataconfig object this class will populate
        self.dataconfig = DataConfig()
        # initialize auxiliary list for allele tracking
        self.alleles = ['V', 'J']

    def _get_reference_pointers(self):
        # aux list
        pointer_to_reference = {'V': self.dataconfig.v_alleles, 'J': self.dataconfig.j_alleles}
        if self.has_d:
            pointer_to_reference['D'] = self.dataconfig.d_alleles
        return pointer_to_reference

    def generate_decaying_probabilities(self, n, base=0.8):
        probabilities = {}
        total_sum = sum(base ** i for i in range(n + 1))
        for i in range(n + 1):
            probabilities[i] = (base ** i) / total_sum
        return probabilities

    def _load_alleles(self, v_alleles, j_alleles, d_alleles=None):
        self.dataconfig.v_alleles = v_alleles
        self.dataconfig.d_alleles = d_alleles
        self.dataconfig.j_alleles = j_alleles

    def _load_gene_usage(self, path):

        with open(path, 'rb') as h:
            self.dataconfig.gene_use_dict = pickle.load(h)

    def _load_trimming_proportions(self, path):
        with open(path, 'rb') as h:
            self.dataconfig.trim_dicts = pickle.load(h)

    def _load_np_lengths(self, path):
        with open(path, 'rb') as h:
            self.dataconfig.NP_lengths = pickle.load(h)

    def _load_np_first_base_use(self, path):
        with open(path, 'rb') as h:
            self.dataconfig.NP_first_bases = pickle.load(h)

    def _load_np_transition_probabilities(self, path):
        with open(path, 'rb') as h:
            self.dataconfig.NP_transitions = pickle.load(h)

    def _derive_3_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        trim_map = dict()
        for t_allele in t_dict.values():
            trim_map[t_allele.name] = dict()
            seq_length = len(t_allele.ungapped_seq)
            for trim_3 in range(seq_length + 1):
                # Trim from the right (3' end)
                trimmed = t_allele.ungapped_seq[:seq_length - trim_3] if trim_3 > 0 else t_allele.ungapped_seq
                trim_map[t_allele.name][seq_length - trim_3] = []
                for v_c_allele in t_dict.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[t_allele.name][seq_length - trim_3].append(v_c_allele.name)
        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_3_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_5_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}

        trim_map = dict()
        for t_allele in t_dict.values():
            trim_map[t_allele.name] = dict()
            for trim_5 in range(len(t_allele.ungapped_seq) + 1):
                trimmed = t_allele.ungapped_seq[trim_5:] if trim_5 > 0 else t_allele.ungapped_seq
                trim_map[t_allele.name][trim_5] = []
                for v_c_allele in t_dict.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[t_allele.name][trim_5].append(v_c_allele.name)
        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_5_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_5_and_3_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        t_list = [i for j in target_alleles for i in target_alleles[j]]
        trim_map = dict()
        for t_allele in t_list:
            trim_map[t_allele.name] = dict()
            for trim_5 in range(len(t_allele.ungapped_seq) + 1):
                for trim_3 in range(len(t_allele.ungapped_seq) - trim_5 + 1):
                    # Correctly handle the trimming for t_allele
                    trimmed = t_allele.ungapped_seq[trim_5:] if trim_5 > 0 else t_allele.ungapped_seq
                    trimmed = trimmed[:-trim_3] if trim_3 > 0 else trimmed

                    trim_map[t_allele.name][(trim_5, trim_3)] = []
                    for d_c_allele in t_list:
                        # Check if the trimmed sequence is a substring of the d_c_allele sequence
                        if trimmed in d_c_allele.ungapped_seq:
                            trim_map[t_allele.name][(trim_5, trim_3)].append(d_c_allele.name)

        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_5_3_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_n_ambiguity_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        comparer = AlleleNComparer()
        for v in t_dict:
            comparer.add_allele(v, t_dict[v].ungapped_seq.upper())

        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_N_AMBIGUITY_CORRECTION_GRAPH'] = comparer

    def make_dataconfig_from_existing_reference_files(self, v_reference_path, j_reference_path, path_to_gene_usage,
                                                      path_to_trimming_proportions, path_to_np_lengths,
                                                      path_to_np_first_base, path_to_np_markov_chain,
                                                      d_reference_path=None,
                                                      ):

        # update d flag
        self.has_d = d_reference_path is not None
        user_d_reference = None
        if self.has_d:
            # add D to aux list to calculate properties for D allele as well
            self.alleles.append('D')

        # 1. read fasta references
        if self.convert_to_asc:
            # ASC logic goes here to resulting variables should be of the following foramt:
            user_v_reference, v_asc_table = create_asc_germline_set(v_reference_path, segment="V")
            # save asc table so reverse transformation will be available to the user
            self.dataconfig.asc_tables['V'] = v_asc_table

            user_j_reference = create_allele_dict(j_reference_path)
            if self.has_d:
                user_d_reference = create_allele_dict(d_reference_path)
        else:

            user_v_reference = create_allele_dict(v_reference_path)
            if d_reference_path is not None:
                user_d_reference = create_allele_dict(d_reference_path)
            user_j_reference = create_allele_dict(j_reference_path)

        print('=' * 50)
        # 2. Fill in Data Config

        # LOAD ALLELES
        self._load_alleles(v_alleles=user_v_reference, d_alleles=user_d_reference, j_alleles=user_j_reference)
        print('Alleles Mounted to DataConfig!...')
        # RANDOM GENE USAGE
        self._load_gene_usage(path_to_gene_usage)
        print('Random Gene Usage Mounted to DataConfig!...')

        # TRIMMING PROPORTIONS
        self._load_trimming_proportions(path_to_trimming_proportions)
        print('Random Trimming Proportions Mounted to DataConfig!...')

        # N REGIONS LENGTHS
        self._load_np_lengths(path_to_np_lengths)
        print('Random NP Region Lengths Mounted to DataConfig!...')

        # N REGIONS  FIRST BASE USAGE
        self._load_np_first_base_use(path_to_np_first_base)
        print('Random NP Initial States Mounted to DataConfig!...')
        # N REGIONS MARKOV TRANSITION MATRICES
        self._load_np_transition_probabilities(path_to_np_markov_chain)
        print('Random NP Markov Chain Mounted to DataConfig!...')

        # 3. Fill in Data Config correction maps
        self._derive_n_ambiguity_map(self.dataconfig.v_alleles)
        print('V Ns Ambiguity Map Mounted to DataConfig!...')

        self._derive_3_prime_correction_map(self.dataconfig.v_alleles)
        print('V 3 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_5_prime_correction_map(self.dataconfig.v_alleles)
        print('V 5 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_3_prime_correction_map(self.dataconfig.j_alleles)
        print('J 3 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_5_prime_correction_map(self.dataconfig.j_alleles)
        print('J 5 Prime Ambiguity Map Mounted to DataConfig!...')
        if self.has_d:
            self._derive_5_and_3_prime_correction_map(self.dataconfig.d_alleles)
            print('D (5,3) Prime Ambiguity Map Mounted to DataConfig!...')

        print('=' * 50)

        return self.dataconfig


class AdaptiveDataConfigGenerator:
    def __init__(self, path_to_base_data_config, convert_to_asc=True):
        self.convert_to_asc = convert_to_asc

        # here we load the dataconfig from which we will estimate the closest matches
        with open(path_to_base_data_config, 'rb') as h:
            self.base_dataconfig = pickle.load(h)

        self.v_asc_table = None
        self.has_d = False
        # initialize an empty dataconfig object this class will populate
        self.dataconfig = DataConfig()
        # initialize auxiliary list for allele tracking
        self.alleles = ['V', 'J']

    def _get_reference_pointers(self):
        # aux list
        pointer_to_reference = {'V': self.dataconfig.v_alleles, 'J': self.dataconfig.j_alleles}
        if self.has_d:
            pointer_to_reference['D'] = self.dataconfig.d_alleles
        return pointer_to_reference

    def generate_decaying_probabilities(self, n, base=0.8):
        probabilities = {}
        total_sum = sum(base ** i for i in range(n + 1))
        for i in range(n + 1):
            probabilities[i] = (base ** i) / total_sum
        return probabilities

    def _load_alleles(self, v_alleles, j_alleles, d_alleles=None):
        self.dataconfig.v_alleles = v_alleles
        self.dataconfig.d_alleles = d_alleles
        self.dataconfig.j_alleles = j_alleles

    def _match_closest_gene_usage(self):


        # TO DO: This is the Random Implementation change this to populate
        # self.dataconfig.gene_use_dict with an updated version with the new alleles based on closest ASC
        pointer_to_reference = self._get_reference_pointers()

        gene_use_dict = {'V': dict(), 'J': dict()}
        if self.has_d:
            gene_use_dict['D'] = dict()

        for allele in self.alleles:
            # get all alleles from reference per gene
            current_alleles = list(pointer_to_reference[allele])
            n = len(current_alleles)
            # uniformly attribute probability to each allele
            random_gene_usage = {i: (1 / n) for i in current_alleles}
            gene_use_dict[allele] = random_gene_usage

        self.dataconfig.gene_use_dict = gene_use_dict

    def _match_closest_trimming_proportions(self):

        # TO DO: This is the Random Implementation change this to populate
        # self.dataconfig.trim_dicts with an updated version with the new alleles based on closest ASC

        pointer_to_reference = self._get_reference_pointers()
        trim_dicts = dict()
        for allele in self.alleles:
            # get only the families
            allele_families = sorted(list(set([i.split('-')[0] for i in pointer_to_reference[allele]])))
            # each family gets up to max_trim trimming amount options with decaying probabilities
            probabilities = self.generate_decaying_probabilities(max_trim)
            trim_dict = {i: probabilities for i in allele_families}

            trim_dicts[allele + '_5'] = trim_dict
            trim_dicts[allele + '_3'] = trim_dict

        self.dataconfig.trim_dicts = trim_dicts

    def _match_closest_np_lengths(self):

        # TO DO: This is the Random Implementation change this to populate
        # self.dataconfig.NP_lengths with an updated version with the new alleles based on closest ASC

        NP_lengths = dict()
        np_regions = ['NP1']
        if self.has_d:
            np_regions.append('NP2')

        for np_region in np_regions:
            probabilities = self.generate_decaying_probabilities(max_size)
            NP_lengths[np_region] = probabilities

        self.dataconfig.NP_lengths = NP_lengths

    def _match_closest_np_first_base_use(self):

        # TO DO: This is the Random Implementation change this to populate
        # self.dataconfig.NP_first_bases with an updated version with the new alleles based on closest ASC

        NP_first_bases = dict()
        np_regions = ['NP1']
        if self.has_d:
            np_regions.append('NP2')

        for np_region in np_regions:
            base_probability = 1 / 4
            NP_first_bases[np_region] = {i: base_probability for i in ['A', 'T', 'C', 'G']}

        self.dataconfig.NP_first_bases = NP_first_bases

    def _match_closest_np_transition_probabilities(self):

        # TO DO: This is the Random Implementation change this to populate
        # self.dataconfig.NP_transitions with an updated version with the new alleles based on closest ASC

        NP_transitions = dict()
        nucleotides = ['A', 'T', 'C', 'G']
        np_regions = ['NP1']
        if self.has_d:
            np_regions.append('NP2')

        for np_region in np_regions:
            NP_transitions[np_region] = dict()
            for position in range(max_size):
                NP_transitions[np_region][position] = dict()
                for base_at_position in nucleotides:
                    # uniform transition probabilities
                    actions = {next_base: 1 / 4 for next_base in nucleotides}
                    NP_transitions[np_region][position][base_at_position] = actions

        self.dataconfig.NP_transitions = NP_transitions

    def _derive_3_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        trim_map = dict()
        for t_allele in t_dict.values():
            trim_map[t_allele.name] = dict()
            seq_length = len(t_allele.ungapped_seq)
            for trim_3 in range(seq_length + 1):
                # Trim from the right (3' end)
                trimmed = t_allele.ungapped_seq[:seq_length - trim_3] if trim_3 > 0 else t_allele.ungapped_seq
                trim_map[t_allele.name][seq_length - trim_3] = []
                for v_c_allele in t_dict.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[t_allele.name][seq_length - trim_3].append(v_c_allele.name)
        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_3_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_5_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}

        trim_map = dict()
        for t_allele in t_dict.values():
            trim_map[t_allele.name] = dict()
            for trim_5 in range(len(t_allele.ungapped_seq) + 1):
                trimmed = t_allele.ungapped_seq[trim_5:] if trim_5 > 0 else t_allele.ungapped_seq
                trim_map[t_allele.name][trim_5] = []
                for v_c_allele in t_dict.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[t_allele.name][trim_5].append(v_c_allele.name)
        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_5_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_5_and_3_prime_correction_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        t_list = [i for j in target_alleles for i in target_alleles[j]]
        trim_map = dict()
        for t_allele in t_list:
            trim_map[t_allele.name] = dict()
            for trim_5 in range(len(t_allele.ungapped_seq) + 1):
                for trim_3 in range(len(t_allele.ungapped_seq) - trim_5 + 1):
                    # Correctly handle the trimming for t_allele
                    trimmed = t_allele.ungapped_seq[trim_5:] if trim_5 > 0 else t_allele.ungapped_seq
                    trimmed = trimmed[:-trim_3] if trim_3 > 0 else trimmed

                    trim_map[t_allele.name][(trim_5, trim_3)] = []
                    for d_c_allele in t_list:
                        # Check if the trimmed sequence is a substring of the d_c_allele sequence
                        if trimmed in d_c_allele.ungapped_seq:
                            trim_map[t_allele.name][(trim_5, trim_3)].append(d_c_allele.name)

        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_5_3_TRIM_SIMILARITY_MAP'] = trim_map

    def _derive_n_ambiguity_map(self, target_alleles):
        t_dict = {i.name: i for j in target_alleles for i in target_alleles[j]}
        comparer = AlleleNComparer()
        for v in t_dict:
            comparer.add_allele(v, t_dict[v].ungapped_seq.upper())

        r_allele = list(t_dict.values())[0]
        allele_ = str(r_allele.type).split('.')[1]  # returns "V" , "D" or "J"
        self.dataconfig.correction_maps[allele_ + '_N_AMBIGUITY_CORRECTION_GRAPH'] = comparer

    def make_dataconfig_from_existing_reference_files(self, v_reference_path, j_reference_path, d_reference_path=None):

        # update d flag
        self.has_d = d_reference_path is not None
        user_d_reference = None
        if self.has_d:
            # add D to aux list to calculate properties for D allele as well
            self.alleles.append('D')

        # 1. read fasta references
        if self.convert_to_asc:
            # ASC logic goes here to resulting variables should be of the following foramt:
            user_v_reference, v_asc_table = create_asc_germline_set(v_reference_path, segment="V")
            # save asc table so reverse transformation will be available to the user
            self.dataconfig.asc_tables['V'] = v_asc_table

            user_j_reference = create_allele_dict(j_reference_path)
            if self.has_d:
                user_d_reference = create_allele_dict(d_reference_path)
        else:

            user_v_reference = create_allele_dict(v_reference_path)
            if d_reference_path is not None:
                user_d_reference = create_allele_dict(d_reference_path)
            user_j_reference = create_allele_dict(j_reference_path)

        print('=' * 50)
        # 2. Fill in Data Config

        # LOAD ALLELES
        self._load_alleles(v_alleles=user_v_reference, d_alleles=user_d_reference, j_alleles=user_j_reference)
        print('Alleles Mounted to DataConfig!...')
        # RANDOM GENE USAGE
        self._match_closest_gene_usage()
        print('Random Gene Usage Mounted to DataConfig!...')

        # TRIMMING PROPORTIONS
        self._match_closest_trimming_proportions()
        print('Random Trimming Proportions Mounted to DataConfig!...')

        # N REGIONS LENGTHS
        self._match_closest_np_lengths()
        print('Random NP Region Lengths Mounted to DataConfig!...')

        # N REGIONS  FIRST BASE USAGE
        self._match_closest_np_first_base_use()
        print('Random NP Initial States Mounted to DataConfig!...')
        # N REGIONS MARKOV TRANSITION MATRICES
        self._match_closest_np_transition_probabilities()
        print('Random NP Markov Chain Mounted to DataConfig!...')

        # ======================================================================= #
        # 3. Fill in Data Config correction maps
        self._derive_n_ambiguity_map(self.dataconfig.v_alleles)
        print('V Ns Ambiguity Map Mounted to DataConfig!...')

        self._derive_3_prime_correction_map(self.dataconfig.v_alleles)
        print('V 3 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_5_prime_correction_map(self.dataconfig.v_alleles)
        print('V 5 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_3_prime_correction_map(self.dataconfig.j_alleles)
        print('J 3 Prime Ambiguity Map Mounted to DataConfig!...')
        self._derive_5_prime_correction_map(self.dataconfig.j_alleles)
        print('J 5 Prime Ambiguity Map Mounted to DataConfig!...')
        if self.has_d:
            self._derive_5_and_3_prime_correction_map(self.dataconfig.d_alleles)
            print('D (5,3) Prime Ambiguity Map Mounted to DataConfig!...')

        print('=' * 50)

        return self.dataconfig
