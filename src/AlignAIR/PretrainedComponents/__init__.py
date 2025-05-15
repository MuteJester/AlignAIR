import pickle
import os

module_dir = os.path.dirname(__file__)

def builtin_orientation_classifier(chain_type='heavy'):
    if chain_type == 'heavy':
        data_path = os.path.join(module_dir, 'Human_IGH_HeavyChain_DNA_Orientation_Pipeline.pkl')
        with open(data_path, 'rb') as h:
            return pickle.load(h)
    elif chain_type == 'tcrb':
        data_path = os.path.join(module_dir, 'Human_TCRB_HeavyChain_DNA_Orientation_Pipeline.pkl')
        with open(data_path, 'rb') as h:
            return pickle.load(h)
    elif chain_type == 'light':
        data_path = os.path.join(module_dir, 'Human_IGH_LightChain_DNA_Orientation_Pipeline.pkl')
        with open(data_path, 'rb') as h:
            return pickle.load(h)
    else:
        raise ValueError(f"Invalid chain type: {chain_type}")