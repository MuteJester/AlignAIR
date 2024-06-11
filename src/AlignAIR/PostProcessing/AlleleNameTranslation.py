import pandas as pd

class TranslateToIMGT:
    def __init__(self,dataconfig):
        self.dataconfig = dataconfig
        #maybe hard code into here the IMGT names, also note that each dataconfig object has the ASC table that
        #can be used to do the translation

        if len(self.dataconfig) == 1: # heavy chain
            self.v_asc_table = self.dataconfig['heavy'].asc_tables['V'].set_index('new_allele')['Allele']
        elif len(self.dataconfig) == 2: # both kappa and lambda of the light chain
            dck = self.dataconfig['kappa'] # kappa chain
            dcl = self.dataconfig['lambda'] # lambda chain
            self.v_asc_table = pd.concat([dck.asc_tables['V'].set_index('new_allele')['Allele'],dcl.asc_tables['V'].set_index('new_allele')['Allele']])

    def translate(self,allele_name):
        return self.v_asc_table[allele_name]

