

class CorrectionDictionaryGenerator:
    def __init__(self,dataconfig,has_d = False):
        self.has_d = has_d
        self.dataconfig = dataconfig


        self.v_alleles = [i for j in dataconfig.v_alleles for i in dataconfig.v_alleles[j]]
        if self.has_d:
            self.d_alleles = [i for j in dataconfig.d_alleles for i in dataconfig.d_alleles[j]]
        self.j_alleles = [i for j in dataconfig.j_alleles for i in dataconfig.j_alleles[j]]

    def _3_prime_5_prime_trim_correction_dictionary(self,alleles):
        trim_map = dict()
        for d_allele in alleles.values():
            trim_map[d_allele.name] = dict()
            for trim_5 in range(len(d_allele.ungapped_seq) + 1):
                for trim_3 in range(len(d_allele.ungapped_seq) - trim_5 + 1):
                    trimmed = d_allele.ungapped_seq[trim_5:] if trim_5 > 0 else d_allele.ungapped_seq
                    trimmed = trimmed[:-trim_3] if trim_3 > 0 else trimmed

                    trim_map[d_allele.name][(trim_5, trim_3)] = []
                    for d_c_allele in alleles.values():
                        if trimmed in d_c_allele.ungapped_seq:
                            trim_map[d_allele.name][(trim_5, trim_3)].append(d_c_allele.name)
        return trim_map

    def _5_prime_trim_correction_dictionary(self,alleles):
        trim_map = dict()
        for v_allele in alleles.values():
            trim_map[v_allele.name] = dict()
            for trim_5 in range(len(v_allele.ungapped_seq) + 1):
                trimmed = v_allele.ungapped_seq[trim_5:] if trim_5 > 0 else v_allele.ungapped_seq
                trim_map[v_allele.name][trim_5] = []
                for v_c_allele in alleles.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[v_allele.name][trim_5].append(v_c_allele.name)
        return trim_map

    def _3_prime_trim_correction_dictionary(self, alleles):
        trim_map = dict()
        for v_allele in alleles.values():
            trim_map[v_allele.name] = dict()
            seq_length = len(v_allele.ungapped_seq)
            for trim_3 in range(seq_length + 1):
                # Trim from the right (3' end)
                trimmed = v_allele.ungapped_seq[:seq_length - trim_3] if trim_3 > 0 else v_allele.ungapped_seq
                trim_map[v_allele.name][seq_length - trim_3] = []
                for v_c_allele in alleles.values():
                    # Check if the trimmed sequence is a substring of the v_c_allele sequence
                    if trimmed in v_c_allele.ungapped_seq:
                        trim_map[v_allele.name][seq_length - trim_3].append(v_c_allele.name)
        return trim_map

