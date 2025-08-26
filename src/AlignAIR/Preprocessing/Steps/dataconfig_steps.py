import pathlib
import pickle

from GenAIRR.dataconfig import DataConfig

from AlignAIR.Data import MultiDataConfigContainer
from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from GenAIRR.data import _CONFIG_NAMES
import GenAIRR.data as data

class ConfigLoadStep(Step):

    def __init__(self, name):
        super().__init__(name)

    import pathlib
    import pickle
    from typing import List

    # Assume these are defined elsewhere in your project
    # from GenAIRR.dataconfig import DataConfig, _CONFIG_NAMES, data

    def read_configs(self,dataconfig_arg: str) -> List['DataConfig']:
        """
        Reads and validates a comma-separated string of data configuration names or file paths.

        Each identifier in the string can be either:
        1. A built-in GenAIRR data configuration name (e.g., 'HUMAN_IGH_OGRDB').
        2. A file path to a pickled DataConfig object.

        Args:
            dataconfig_arg: A string containing one or more config identifiers, separated by commas.

        Returns:
            A list of loaded DataConfig objects.

        Raises:
            ValueError: If an identifier is invalid, not found, or if a file cannot be
                        correctly loaded and verified as a DataConfig object.
        """
        if not isinstance(dataconfig_arg, str) or not dataconfig_arg:
            raise ValueError("Input must be a non-empty string.")

        config_identifiers = dataconfig_arg.split(',')
        loaded_configs = []

        for identifier in config_identifiers:
            identifier = identifier.strip()  # Handle whitespace
            config_path = pathlib.Path(identifier)

            # Case 1: The identifier is a path to an existing file
            if config_path.is_file():
                try:
                    with open(config_path, 'rb') as f:
                        loaded_obj = pickle.load(f)

                    if not isinstance(loaded_obj, DataConfig):
                        raise TypeError(f"Object in {identifier} is not a DataConfig instance.")

                    loaded_configs.append(loaded_obj)

                except pickle.UnpicklingError:
                    raise ValueError(f"File '{identifier}' is not a valid pickle file.")
                except Exception as e:
                    raise ValueError(f"Failed to load file '{identifier}'. Error: {e}")

            # Case 2: The identifier is a built-in config name
            elif identifier in _CONFIG_NAMES:
                try:
                    p_dataconfig = getattr(data, identifier)
                    loaded_configs.append(p_dataconfig)
                except AttributeError:
                    raise ValueError(f"Could not load built-in config '{identifier}'.")

            # Case 3: The identifier is not valid
            else:
                raise ValueError(
                    f"Invalid data configuration: '{identifier}'. Must be a valid GenAIRR "
                    f"data config name or a path to a DataConfig .pkl file."
                )

        return loaded_configs

    def process(self, predict_object: PredictObject):
        """
        Loads configuration based on the chain type from provided paths.
        Supports both new genairr_dataconfig parameter and legacy individual config parameters.

        Args:
            predict_object (PredictObject): The object that holds the chain_type and config_paths.

        Returns:
            PredictObject: Updated with loaded configuration.
        """
        args = predict_object.script_arguments

        self.log(f"Loading Data Config")

        # Handle legacy parameters for backward compatibility
        if hasattr(args, 'genairr_dataconfig') and args.genairr_dataconfig:
            config_arg = args.genairr_dataconfig
        else:
            # Legacy mode: build config string from individual parameters
            legacy_configs = []
            
            if hasattr(args, 'heavy_data_config') and args.heavy_data_config and args.heavy_data_config != 'D':
                legacy_configs.append(args.heavy_data_config)
            elif hasattr(args, 'chain_type') and args.chain_type == 'heavy':
                legacy_configs.append('HUMAN_IGH_OGRDB')
                
            if hasattr(args, 'lambda_data_config') and args.lambda_data_config and args.lambda_data_config != 'D':
                legacy_configs.append(args.lambda_data_config)
            elif hasattr(args, 'chain_type') and args.chain_type == 'light':
                legacy_configs.append('HUMAN_IGL_OGRDB')
                
            if hasattr(args, 'kappa_data_config') and args.kappa_data_config and args.kappa_data_config != 'D':
                legacy_configs.append(args.kappa_data_config)
            elif hasattr(args, 'chain_type') and args.chain_type == 'light':
                if 'HUMAN_IGL_OGRDB' not in legacy_configs:  # Avoid duplicates
                    legacy_configs.append('HUMAN_IGK_OGRDB')
            
            if not legacy_configs:
                # Fallback to default
                legacy_configs = ['HUMAN_IGH_OGRDB']
                
            config_arg = ','.join(legacy_configs)
            self.log(f"Using legacy config mode: {config_arg}")

        loaded_configs = self.read_configs(config_arg)
        predict_object.dataconfig = MultiDataConfigContainer(loaded_configs)

        predict_object.mount_genotype_list()
        self.log("Data Config loaded successfully, Config Mounted : {}".format(config_arg))

        return predict_object

