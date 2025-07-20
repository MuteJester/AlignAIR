import pathlib
import pickle
from typing import List, Union, Dict

from GenAIRR.dataconfig import DataConfig


# Assume these are defined elsewhere in your project
# from GenAIRR.dataconfig import DataConfig, _CONFIG_NAMES, data

class MultiDataConfigContainer:
    def packaged_config(self) -> Dict[str, DataConfig]:
        """
        Return a dictionary mapping chain types to their DataConfig objects.
        For single-chain, returns {chain_type: DataConfig}.
        For multi-chain, returns {chain_type: DataConfig, ...}.
        """
        result = {}
        for dc in self.dataconfigs:
            chain_type = getattr(dc.metadata, 'chain_type', None)
            if chain_type is None:
                raise ValueError(f"DataConfig missing chain_type in metadata: {dc}")
            result[chain_type] = dc
        return result
    """
    A container for one or more DataConfig objects.

    If it holds a single DataConfig, it acts as a proxy to that object,
    allowing direct attribute access (e.g., container.v_alleles).

    If it holds multiple DataConfigs, it acts as a list-like object,
    requiring iteration or indexing to access individual configs
    (e.g., for dc in container: ... or container[0].v_alleles).
    """

    def __init__(self, dataconfigs: List[DataConfig]):
        if not dataconfigs:
            raise ValueError("MultiDataConfigContainer cannot be initialized with an empty list.")
        self.dataconfigs = dataconfigs

    def all_have_d(self) -> bool:
        """
        Check if all contained DataConfigs have D alleles.
        """
        return all(dc.metadata.has_d for dc in self.dataconfigs)

    def has_at_least_one_d(self) -> bool:
        """
        Check if at least one contained DataConfig has D alleles.
        """
        return any(dc.metadata.has_d for dc in self.dataconfigs)

    def chain_types(self) -> List[str]:
        """
        Get the chain types of all contained DataConfigs.
        """
        return [dc.metadata.chain_type for dc in self.dataconfigs]

    @property
    def number_of_chain_types(self) -> int:
        """
        Get the number of unique chain types across all DataConfigs.

        Returns:
            int: Number of unique chain types
        """
        return len(set(self.chain_types()))

    def derive_combined_allele_dictionaries(self) -> Dict[str, Dict[str, any]]:
        """
        Derive combined allele dictionaries from all DataConfigs.
        Combines alleles across all configs for each gene type.
        
        Returns:
            Dict with keys 'v_dict', 'j_dict', and optionally 'd_dict'
        """
        combined_dicts = {}
        genes_to_process = ['v', 'j']
        if self.has_at_least_one_d():
            genes_to_process.append('d')
            
        for gene in genes_to_process:
            combined_dict = {}
            for dc in self.dataconfigs:
                if hasattr(dc, f'{gene}_alleles'):
                    aux_dict = {i.name: i for i in dc.allele_list(gene)}
                    aux_dict = dict(sorted(aux_dict.items()))
                    combined_dict.update(aux_dict)
            combined_dicts[f'{gene}_dict'] = combined_dict
            
        return combined_dicts

    def get_allele_counts(self) -> Dict[str, int]:
        """
        Get combined allele counts across all DataConfigs.
        
        Returns:
            Dict with allele counts for each gene type
        """
        counts = {
            'v_allele_count': self.number_of_v_alleles,
            'j_allele_count': self.number_of_j_alleles,
        }
        
        if self.has_at_least_one_d():
            counts['d_allele_count'] = self.number_of_j_alleles
                
        return counts

    def has_missing_d(self) -> bool:
        """
        Check if there's heterogeneous D presence (some configs have D, others don't).
        Returns False if either all configs have D or none have D.
        
        Returns:
            bool: True if mixed D presence, False if uniform
        """
        configs_with_d = sum(1 for dc in self.dataconfigs if dc.metadata.has_d)
        total_configs = len(self.dataconfigs)
        
        # Return True only if some (but not all) configs have D
        return 0 < configs_with_d < total_configs

    def equal_batch_partitioning(self, total_batch_size: int) -> List[int]:
        n_configs = len(self.dataconfigs)
        base_size = total_batch_size // n_configs
        remainder = total_batch_size % n_configs

        batch_sizes = []
        for i in range(n_configs):
            # Distribute remainder evenly across configs
            size = base_size + (1 if i < remainder else 0)
            batch_sizes.append(size)

        # Ensure total equals expected batch size
        assert sum(
            batch_sizes) == total_batch_size, f"Batch partitioning error: {sum(batch_sizes)} != {total_batch_size}"

        return batch_sizes

    @property
    def number_of_v_alleles(self) -> int:
        """
        Get the total number of unique V alleles across all DataConfigs.
        
        Returns:
            int: Combined count of V alleles
        """
        combined_dicts = self.derive_combined_allele_dictionaries()
        return len(combined_dicts.get('v_dict', {}))

    @property
    def number_of_d_alleles(self) -> int:
        """
        Get the total number of unique D alleles across all DataConfigs.
        Includes the "Short-D" allele for compatibility.

        Returns:
            int: Combined count of D alleles (including Short-D)
        """
        if not self.has_at_least_one_d():
            return 0
        combined_dicts = self.derive_combined_allele_dictionaries()
        d_dict_size = len(combined_dicts.get('d_dict', {}))

        # Add 1 for 'Short-D' if not already in the dictionary
        if 'Short-D' not in combined_dicts.get('d_dict', {}):
            d_dict_size += 1

        return d_dict_size
    @property
    def number_of_j_alleles(self) -> int:
        """
        Get the total number of unique J alleles across all DataConfigs.
        
        Returns:
            int: Combined count of J alleles
        """
        combined_dicts = self.derive_combined_allele_dictionaries()
        return len(combined_dicts.get('j_dict', {}))

    def __getattr__(self, name: str):
        """
        Proxy attribute access to the single DataConfig object if only one exists.
        """
        # Use object.__getattribute__ to safely access 'dataconfigs' and prevent recursion.
        dataconfigs = object.__getattribute__(self, 'dataconfigs')

        if len(dataconfigs) == 1:
            try:
                # Forward the attribute access to the single DataConfig object.
                return getattr(dataconfigs[0], name)
            except AttributeError:
                # The attribute doesn't exist on the underlying DataConfig object.
                raise AttributeError(f"'{type(dataconfigs[0]).__name__}' object has no attribute '{name}'")

        # If there are multiple configs, accessing attributes directly is not allowed.
        # We can safely use len(self) here because __len__ doesn't trigger __getattr__.
        raise AttributeError(
            f"'{type(self).__name__}' object with {len(self)} items has no attribute '{name}'. "
            "Access configs by iterating or indexing (e.g., container[0].attribute)."
        )

    def __iter__(self):
        """Allows iterating over the contained DataConfig objects."""
        return iter(self.dataconfigs)

    def __getitem__(self, index: int) -> 'DataConfig':
        """
        Get a DataConfig by index.
        """
        return self.dataconfigs[index]

    def __len__(self) -> int:
        """
        Get the number of DataConfigs in the container.
        """
        return len(self.dataconfigs)
