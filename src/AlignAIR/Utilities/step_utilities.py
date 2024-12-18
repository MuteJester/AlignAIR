from GenAIRR.data import builtin_kappa_chain_data_config, builtin_lambda_chain_data_config, \
    builtin_heavy_chain_data_config
import pickle
import pathlib

from AlignAIR.Data import HeavyChainDataset, LightChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR


class DataConfigLibrary:
    def __init__(self, custom_heavy_data_config=None, custom_kappa_data_config=None, custom_lambda_data_config=None):
        self.data_configs = {'heavy': builtin_heavy_chain_data_config(),
                             'kappa': builtin_kappa_chain_data_config(),
                             'lambda': builtin_lambda_chain_data_config()
                             }

        self.mount = None
        if custom_heavy_data_config != 'D':
            with open(custom_heavy_data_config, 'rb') as h:
                self.data_configs['heavy'] = pickle.load(h)
        if custom_kappa_data_config != 'D':
            with open(custom_kappa_data_config, 'rb') as h:
                self.data_configs['kappa'] = pickle.load(h)
        if custom_lambda_data_config != 'D':
            with open(custom_lambda_data_config, 'rb') as h:
                self.data_configs['lambda'] = pickle.load(h)

    @property
    def mounted(self):
        return self.mount

    @property
    def heavy_(self):
        return self.data_configs['heavy']

    @property
    def kappa_(self):
        return self.data_configs['kappa']

    @property
    def lambda_(self):
        return self.data_configs['lambda']

    def mount_type(self, chain_type):
        self.mount = chain_type

    def config(self, sub_type=None):
        if self.mount == 'heavy':
            return self.data_configs['heavy']
        else:
            return self.data_configs[sub_type]

    def packaged_config(self):
        if self.mount == 'heavy':
            return {'heavy': self.data_configs['heavy']}
        elif self.mount == 'light':
            return {'kappa': self.data_configs['kappa'], 'lambda': self.data_configs['lambda']}
        else:
            raise ValueError('No Chain Type Mounted / Unknown Chain Type {}'.format(self.mount))

    @property
    def matching_dataset_object(self):
        if self.mounted == 'heavy':
            return HeavyChainDataset
        elif self.mounted == 'light':
            return LightChainDataset
        else:
            raise ValueError('No Chain Type Mounted / Unknown Chain Type {}'.format(self.mounted))

    @property
    def matching_alignair_model(self):
        if self.mounted == 'heavy':
            return HeavyChainAlignAIRR
        elif self.mounted == 'light':
            return LightChainAlignAIRR
        else:
            raise ValueError('No Chain Type Mounted / Unknown Chain Type {}'.format(self.mounted))

    def reference_allele_sequences(self, allele: str):
        """
        Get the reference allele sequences for the given allele type.
        Args:
            allele:

        Returns:

        """
        if allele not in ['v', 'd', 'j']:
            raise ValueError("Allele must be one of 'v', 'd', or 'j'")

        if self.mounted == 'heavy':
            allele_dict = getattr(self.config(), f'{allele}_alleles')
            return [i.ungapped_seq.upper() for j in allele_dict for i in allele_dict[j]]
        elif self.mounted == 'light':
            if allele == 'd':
                return []

            allele_dict = getattr(self.config('kappa'), f'{allele}_alleles')
            alleles_kappa = [i.ungapped_seq.upper() for j in allele_dict for i in allele_dict[j]]
            allele_dict = getattr(self.config('lambda'), f'{allele}_alleles')
            alleles_lambda = [i.ungapped_seq.upper() for j in allele_dict for i in allele_dict[j]]
            return alleles_kappa + alleles_lambda
        else:
            raise ValueError('No Chain Type Mounted / Unknown Chain Type {}'.format(self.mounted))


class FileInfo:
    def __init__(self, path):
        self.path = path
        self.file_name, self.file_suffix = self.get_filename(path)
        self.file_type = self.file_suffix.replace('.', '')
        self.sample_count = 0

    def set_sample_count(self, sample_count):
        self.sample_count = sample_count

    def __len__(self):
        return self.sample_count

    @staticmethod
    def get_filename(path, return_suffix=True):
        path_object = pathlib.Path(path)
        if return_suffix:
            return path_object.stem, path_object.suffix
        else:
            return path_object.stem
