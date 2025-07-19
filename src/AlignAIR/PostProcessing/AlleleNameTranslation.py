import pandas as pd

class TranslateToIMGT:
    def __init__(self, dataconfig):
        self.dataconfig = dataconfig
        # dataconfig is a dict mapping chain_type to DataConfig
        # Build a combined ASC table for V alleles from all chains
        v_asc_tables = []
        for chain, dc in self.dataconfig.items():
            if hasattr(dc, 'asc_tables') and 'V' in dc.asc_tables:
                v_asc = dc.asc_tables['V']
                v_asc_tables.append(v_asc.set_index('new_allele')['imgt_allele'])
        if v_asc_tables:
            self.v_asc_table = pd.concat(v_asc_tables)
        else:
            self.v_asc_table = pd.Series(dtype=object)

    def translate(self, allele_name):
        # Only translate V alleles using the combined ASC table
        if "V" in allele_name:
            try:
                asc_alleles = self.v_asc_table[allele_name]
                asc_alleles = ','.join(asc_alleles) if not isinstance(asc_alleles, str) else asc_alleles
                return asc_alleles
            except KeyError:
                # If allele not found, return original name
                return allele_name
        else:
            return allele_name

