import pandas as pd
from tqdm.auto import tqdm
from VDeepJDataPrepper import VDeepJDataPrepper

class VDeepJDataset:
    def __init__(self, data_path, max_sequence_length=512,nrows=None):
        self.max_sequence_length = max_sequence_length
        self.data_prepper = VDeepJDataPrepper(self.max_sequence_length)
        self.data = pd.read_table(data_path,
                                  usecols=['sequence', 'v_call', 'd_call', 'j_call', 'v_sequence_end',
                                           'd_sequence_start',
                                           'j_sequence_start',
                                           'j_sequence_end', 'd_sequence_end', 'v_sequence_start'],nrows=nrows)


        self.derive_call_dictionaries()
        self.derive_call_sections()
        self.derive_call_one_hot_representation()
        self.derive_counts()

    def derive_counts(self):
        self.v_family_count = len(self.v_family_call_ohe)
        self.v_gene_count = len(self.v_gene_call_ohe)
        self.v_allele_count = len(self.v_allele_call_ohe)

        self.d_family_count = len(self.d_family_call_ohe)
        self.d_gene_count = len(self.d_gene_call_ohe)
        self.d_allele_count = len(self.d_allele_call_ohe)

        self.j_gene_count = len(self.j_gene_call_ohe)
        self.j_allele_count = len(self.j_allele_call_ohe)

    def derive_call_one_hot_representation(self):
        self.v_family_call_ohe, self.v_family_call_ohe_np, self.v_gene_call_ohe, \
        self.v_gene_call_ohe_np, self.v_allele_call_ohe, \
        self.v_allele_call_ohe_np = self.data_prepper.convert_calls_to_one_hot(self.v_gene,
                                                                               self.v_allele,
                                                                               self.v_family)

        self.d_family_call_ohe, self.d_family_call_ohe_np, \
        self.d_gene_call_ohe, self.d_gene_call_ohe_np, \
        self.d_allele_call_ohe, self.d_allele_call_ohe_np = \
            self.data_prepper.convert_calls_to_one_hot(self.d_gene,
                                                       self.d_allele,
                                                       self.d_family)

        self.j_gene_call_ohe, self.j_gene_call_ohe_np, self.j_allele_call_ohe, self.j_allele_call_ohe_np = \
            self.data_prepper.convert_calls_to_one_hot(self.j_gene, self.j_allele)

    def derive_call_sections(self):
        self.v_family = [self.v_dict[x]['family'] for x in tqdm(self.data.v_call)]
        self.d_family = [self.d_dict[x]['family'] for x in tqdm(self.data.d_call)]

        self.v_gene = [self.v_dict[x]['gene'] for x in tqdm(self.data.v_call)]
        self.d_gene = [self.d_dict[x]['gene'] for x in tqdm(self.data.d_call)]
        self.j_gene = [self.j_dict[x]['gene'] for x in tqdm(self.data.j_call)]

        self.v_allele = [self.v_dict[x]['allele'] for x in tqdm(self.data.v_call)]
        self.d_allele = [self.d_dict[x]['allele'] for x in tqdm(self.data.d_call)]
        self.j_allele = [self.j_dict[x]['allele'] for x in tqdm(self.data.j_call)]

    def derive_call_dictionaries(self):
        self.v_dict, self.d_dict, self.j_dict = \
            self.data_prepper.get_family_gene_allele_map(self.data.v_call.unique(),
                                                         self.data.d_call.unique(),
                                                         self.data.j_call.unique())

    def get_train_generator(self,batch_size,corrupt_beginning=False):
        return self.data_prepper.get_train_dataset(self.data,
                                                   self.v_family_call_ohe_np,
                                                   self.v_gene_call_ohe_np,
                                                   self.v_allele_call_ohe_np,
                                                   self.d_family_call_ohe_np,
                                                   self.d_gene_call_ohe_np,
                                                   self.d_allele_call_ohe_np,
                                                   self.j_gene_call_ohe_np,
                                                   self.j_allele_call_ohe_np,
                                                   batch_size,
                                                   train=True,
                                                   corrupt_beginning=corrupt_beginning)

    def get_eval_generator(self,batch_size):
        return self.data_prepper.get_train_dataset(self.data,
                                                   self.v_family_call_ohe_np,
                                                   self.v_gene_call_ohe_np,
                                                   self.v_allele_call_ohe_np,
                                                   self.d_family_call_ohe_np,
                                                   self.d_gene_call_ohe_np,
                                                   self.d_allele_call_ohe_np,
                                                   self.j_gene_call_ohe_np,
                                                   self.j_allele_call_ohe_np,
                                                   batch_size,
                                                   train=False,
                                                   corrupt_beginning=False)

    def generate_model_params(self):
        return {
            'max_seq_length':self.max_sequence_length,
            'v_family_count':self.v_family_count,
            'v_gene_count':self.v_gene_count,
            'v_allele_count':self.v_allele_count,
            'd_family_count':self.d_family_count,
            'd_gene_count':self.d_gene_count,
            'd_allele_count':self.d_allele_count,
            'j_gene_count':self.j_gene_count,
            'j_allele_count':self.j_allele_count
        }

    def __len__(self):
        return len(self.data)


