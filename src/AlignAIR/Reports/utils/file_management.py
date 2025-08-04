import pickle


def load_pickle(file_path):
    """Loads a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_dataconfig(dataconfig_name_or_path):
    """Loads a DataConfig from a built-in name or a .pkl file."""
    import GenAIRR.data as genairr_data
    from GenAIRR.dataconfig import DataConfig
    from pathlib import Path

    if Path(dataconfig_name_or_path).is_file():
        with open(dataconfig_name_or_path, 'rb') as f:
            return pickle.load(f)
    elif hasattr(genairr_data, dataconfig_name_or_path):
        return getattr(genairr_data, dataconfig_name_or_path)
    else:
        raise ValueError(f"Invalid dataconfig: '{dataconfig_name_or_path}'.")
