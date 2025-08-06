import pickle

from AlignAIR.Data import MultiDataConfigContainer


def load_pickle(file_path):
    """Loads a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_dataconfig(dataconfig_name_or_path):
    """Loads a DataConfig from a built-in name or a .pkl file."""
    import GenAIRR.data as genairr_data
    from GenAIRR.dataconfig import DataConfig
    from pathlib import Path

    configs = dataconfig_name_or_path.split(',')
    if len (configs) == 1:
        if Path(dataconfig_name_or_path).is_file():
            with open(dataconfig_name_or_path, 'rb') as f:
                return pickle.load(f)
        elif hasattr(genairr_data, dataconfig_name_or_path):
            return getattr(genairr_data, dataconfig_name_or_path)
        else:
            raise ValueError(f"Invalid dataconfig: '{dataconfig_name_or_path}'.")
    else:
        dataconfigs = []
        for config in configs:
            if Path(config).is_file():
                with open(config, 'rb') as f:
                    dataconfigs.append(pickle.load(f))
            elif hasattr(genairr_data, config):
                dataconfigs.append(getattr(genairr_data, config))
            else:
                raise ValueError(f"Invalid dataconfig: '{config}'.")
        return MultiDataConfigContainer(dataconfigs)
