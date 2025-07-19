import pathlib

from GenAIRR.data import _CONFIG_NAMES
for config in _CONFIG_NAMES:
    if config not in globals():
        globals()[config] = __import__(f"GenAIRR.data", fromlist=[config])

from AlignAIR.Data import SingleChainDataset,MultiChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR
from dataclasses import dataclass
import pickle
import os

# ─────────────────────────────────────────────────────────────────────────────
# Registry definitions for built-ins, datasets, and model mapping
# ─────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
#
# @dataclass
# class ChainConfig:
#     types: list = ('heavy', 'kappa', 'lambda', 'tcrb')
#     builtin_loaders: dict = None
#     datasets: dict = None
#     models: dict = None
#
#     def __post_init__(self):
#         self.builtin_loaders = {
#             'heavy': builtin_heavy_chain_data_config,
#             'kappa': builtin_kappa_chain_data_config,
#             'lambda': builtin_lambda_chain_data_config,
#             'tcrb': builtin_tcrb_data_config,
#             'heavy_extended': builtin_heavy_chain_data_config,
#         }
#         self.datasets = {
#             'heavy': HeavyChainDataset,
#             'light': LightChainDataset,
#             'tcrb': HeavyChainDataset,
#         }
#         self.models = {
#             'heavy': HeavyChainAlignAIRR,
#             'light': LightChainAlignAIRR,
#             'tcrb': HeavyChainAlignAIRR,
#         }
#
# class DataConfigLibrary:
#     def __init__(self, custom_data_configs=None, chain_config=None):
#         self.chain_config = chain_config or ChainConfig()
#         self.data_configs = {}
#         self.mount = None
#         custom_data_configs = custom_data_configs or {}
#
#         for chain in self.chain_config.types:
#             custom_path = custom_data_configs.get(chain, 'D')
#             if custom_path == 'D':
#                 self.data_configs[chain] = self.chain_config.builtin_loaders[chain]()
#             else:
#                 with open(custom_path, 'rb') as f:
#                     self.data_configs[chain] = pickle.load(f)
#
#     # ───── Public API ─────
#
#     @property
#     def mounted(self):
#         return self.mount
#
#     def mount_type(self, chain_type):
#         if chain_type not in self.data_configs and chain_type != 'light':
#             raise ValueError(f"Unknown chain type: {chain_type}")
#         self.mount = chain_type
#
#     def config(self, sub_type=None):
#         if self.mount and self.mount != 'light':
#             return self.data_configs[self.mount]
#         elif sub_type:
#             return self.data_configs[sub_type]
#         else:
#             raise ValueError("No chain mounted and no subtype provided.")
#
#     def packaged_config(self):
#         if self.mounted == 'heavy':
#             return {'heavy': self.data_configs['heavy']}
#         elif self.mounted == 'light':
#             return {
#                 'kappa': self.data_configs['kappa'],
#                 'lambda': self.data_configs['lambda']
#             }
#         elif self.mounted == 'tcrb':
#             return {'tcrb': self.data_configs['tcrb']}
#         else:
#             raise ValueError(f"Unsupported mounted chain type: {self.mounted}")
#
#     @property
#     def matching_dataset_object(self):
#         if not self.mounted:
#             raise ValueError("No Chain Type Mounted.")
#         return self.chain_config.datasets[self.mounted]
#
#     @property
#     def matching_alignair_model(self):
#         if not self.mounted:
#             raise ValueError("No Chain Type Mounted.")
#         return self.chain_config.models[self.mounted]
#
#     # ───── Allele Handling ─────
#
#     def _get_allele_dicts(self, allele: str):
#         if allele not in ['v', 'd', 'j']:
#             raise ValueError("Allele must be one of 'v', 'd', or 'j'")
#
#         if self.mounted in ['heavy', 'tcrb']:
#             return [getattr(self.config(), f'{allele}_alleles')]
#         elif self.mounted == 'light':
#             if allele == 'd':
#                 return []
#             return [
#                 getattr(self.config('kappa'), f'{allele}_alleles'),
#                 getattr(self.config('lambda'), f'{allele}_alleles')
#             ]
#         else:
#             raise ValueError(f"Unsupported mounted chain type: {self.mounted}")
#
#     def reference_allele_sequences(self, allele: str):
#         dicts = self._get_allele_dicts(allele)
#         return [i.ungapped_seq.upper() for d in dicts for j in d for i in d[j]]
#
#     def reference_allele_names(self, allele: str):
#         dicts = self._get_allele_dicts(allele)
#         return [i.name for d in dicts for j in d for i in d[j]]
#
#     def get_allele_dict(self, allele: str):
#         dicts = self._get_allele_dicts(allele)
#         result = {}
#         for d in dicts:
#             result.update({i.name: i.ungapped_seq.upper() for j in d for i in d[j]})
#         return result
#
#     @property
#     def heavy_(self):
#         return self.data_configs['heavy']
#
#     @property
#     def kappa_(self):
#         return self.data_configs['kappa']
#
#     @property
#     def lambda_(self):
#         return self.data_configs['lambda']
#
#     @property
#     def tcrb_(self):
#         return self.data_configs['tcrb']

import pathlib
from typing import List


class FileInfo:
    """
    Holds information about a single file, including its path, name, type, and sample count.
    """

    def __init__(self, path: str):
        if not path or not isinstance(path, str):
            raise ValueError("File path must be a non-empty string.")

        self.path = path
        path_object = pathlib.Path(path)
        self.file_name = path_object.stem
        self.file_suffix = path_object.suffix
        self.file_type = self.file_suffix.lstrip('.')
        self.sample_count = 0

    def set_sample_count(self, sample_count: int):
        """Sets the number of samples (e.g., rows) in the file."""
        self.sample_count = sample_count

    def __len__(self) -> int:
        """Returns the number of samples in the file."""
        return self.sample_count

    def __repr__(self) -> str:
        return f"FileInfo(path='{self.path}', name='{self.file_name}', type='{self.file_type}')"


class MultiFileInfoContainer:
    """
    A container for one or more FileInfo objects, created from a comma-separated path string.

    If it holds a single FileInfo object, it acts as a proxy, allowing direct attribute
    access (e.g., container.file_name).

    If it holds multiple FileInfo objects, it acts as a list-like object,
    requiring iteration or indexing to access individual file info
    (e.g., for fi in container: ... or container[0].file_name).
    """

    def __init__(self, paths_arg: str):
        if not paths_arg or not isinstance(paths_arg, str):
            raise ValueError("Input must be a non-empty, comma-separated string of file paths.")

        paths = [p.strip() for p in paths_arg.split(',') if p.strip()]
        if not paths:
            raise ValueError("Input string contains no valid file paths.")

        self.file_infos = [FileInfo(p) for p in paths]

    def file_names(self) -> List[str]:
        """Returns a list of all file names."""
        return [fi.file_name for fi in self.file_infos]

    def file_types(self) -> List[str]:
        """Returns a list of all file types."""
        return [fi.file_type for fi in self.file_infos]

    def total_sample_count(self) -> int:
        """Returns the sum of sample counts from all files."""
        return sum(fi.sample_count for fi in self.file_infos)

    def __getattr__(self, name: str):
        """
        Proxy attribute access to the single FileInfo object if only one exists.
        """
        file_infos = object.__getattribute__(self, 'file_infos')
        if len(file_infos) == 1:
            try:
                return getattr(file_infos[0], name)
            except AttributeError:
                raise AttributeError(f"'{type(file_infos[0]).__name__}' object has no attribute '{name}'")

        raise AttributeError(
            f"'{type(self).__name__}' object with {len(self)} items has no attribute '{name}'. "
            "Access file info by iterating or indexing (e.g., container[0].attribute)."
        )

    def __iter__(self):
        """Allows iterating over the contained FileInfo objects."""
        return iter(self.file_infos)

    def __getitem__(self, index: int) -> FileInfo:
        """Get a FileInfo object by index."""
        return self.file_infos[index]

    def __len__(self) -> int:
        """Get the number of FileInfo objects in the container."""
        return len(self.file_infos)
