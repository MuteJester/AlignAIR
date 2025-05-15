import pathlib

from GenAIRR.data import (
    builtin_heavy_chain_data_config,
    builtin_kappa_chain_data_config,
    builtin_lambda_chain_data_config,
    builtin_tcrb_data_config,
)
from AlignAIR.Data import HeavyChainDataset, LightChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR
from dataclasses import dataclass
import pickle
import os

# ─────────────────────────────────────────────────────────────────────────────
# Registry definitions for built-ins, datasets, and model mapping
# ─────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass

@dataclass
class ChainConfig:
    types: list = ('heavy', 'kappa', 'lambda', 'tcrb')
    builtin_loaders: dict = None
    datasets: dict = None
    models: dict = None

    def __post_init__(self):
        self.builtin_loaders = {
            'heavy': builtin_heavy_chain_data_config,
            'kappa': builtin_kappa_chain_data_config,
            'lambda': builtin_lambda_chain_data_config,
            'tcrb': builtin_tcrb_data_config,
        }
        self.datasets = {
            'heavy': HeavyChainDataset,
            'light': LightChainDataset,
            'tcrb': HeavyChainDataset,
        }
        self.models = {
            'heavy': HeavyChainAlignAIRR,
            'light': LightChainAlignAIRR,
            'tcrb': HeavyChainAlignAIRR,
        }

class DataConfigLibrary:
    def __init__(self, custom_data_configs=None, chain_config=None):
        self.chain_config = chain_config or ChainConfig()
        self.data_configs = {}
        self.mount = None
        custom_data_configs = custom_data_configs or {}

        for chain in self.chain_config.types:
            custom_path = custom_data_configs.get(chain, 'D')
            if custom_path == 'D':
                self.data_configs[chain] = self.chain_config.builtin_loaders[chain]()
            else:
                with open(custom_path, 'rb') as f:
                    self.data_configs[chain] = pickle.load(f)

    # ───── Public API ─────

    @property
    def mounted(self):
        return self.mount

    def mount_type(self, chain_type):
        if chain_type not in self.data_configs and chain_type != 'light':
            raise ValueError(f"Unknown chain type: {chain_type}")
        self.mount = chain_type

    def config(self, sub_type=None):
        if self.mount and self.mount != 'light':
            return self.data_configs[self.mount]
        elif sub_type:
            return self.data_configs[sub_type]
        else:
            raise ValueError("No chain mounted and no subtype provided.")

    def packaged_config(self):
        if self.mounted == 'heavy':
            return {'heavy': self.data_configs['heavy']}
        elif self.mounted == 'light':
            return {
                'kappa': self.data_configs['kappa'],
                'lambda': self.data_configs['lambda']
            }
        elif self.mounted == 'tcrb':
            return {'tcrb': self.data_configs['tcrb']}
        else:
            raise ValueError(f"Unsupported mounted chain type: {self.mounted}")

    @property
    def matching_dataset_object(self):
        if not self.mounted:
            raise ValueError("No Chain Type Mounted.")
        return self.chain_config.datasets[self.mounted]

    @property
    def matching_alignair_model(self):
        if not self.mounted:
            raise ValueError("No Chain Type Mounted.")
        return self.chain_config.models[self.mounted]

    # ───── Allele Handling ─────

    def _get_allele_dicts(self, allele: str):
        if allele not in ['v', 'd', 'j']:
            raise ValueError("Allele must be one of 'v', 'd', or 'j'")

        if self.mounted in ['heavy', 'tcrb']:
            return [getattr(self.config(), f'{allele}_alleles')]
        elif self.mounted == 'light':
            if allele == 'd':
                return []
            return [
                getattr(self.config('kappa'), f'{allele}_alleles'),
                getattr(self.config('lambda'), f'{allele}_alleles')
            ]
        else:
            raise ValueError(f"Unsupported mounted chain type: {self.mounted}")

    def reference_allele_sequences(self, allele: str):
        dicts = self._get_allele_dicts(allele)
        return [i.ungapped_seq.upper() for d in dicts for j in d for i in d[j]]

    def reference_allele_names(self, allele: str):
        dicts = self._get_allele_dicts(allele)
        return [i.name for d in dicts for j in d for i in d[j]]

    def get_allele_dict(self, allele: str):
        dicts = self._get_allele_dicts(allele)
        result = {}
        for d in dicts:
            result.update({i.name: i.ungapped_seq.upper() for j in d for i in d[j]})
        return result

    @property
    def heavy_(self):
        return self.data_configs['heavy']

    @property
    def kappa_(self):
        return self.data_configs['kappa']

    @property
    def lambda_(self):
        return self.data_configs['lambda']

    @property
    def tcrb_(self):
        return self.data_configs['tcrb']


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
