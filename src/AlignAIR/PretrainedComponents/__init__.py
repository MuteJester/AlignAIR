import pickle
import os
from GenAIRR.dataconfig.enums import ChainType
module_dir = os.path.dirname(__file__)

def builtin_orientation_classifier(chain_type:ChainType = ChainType.BCR_HEAVY):
    if chain_type == ChainType.BCR_HEAVY:
        data_path = os.path.join(module_dir, 'Human_IGH_HeavyChain_DNA_Orientation_Pipeline.pkl')
        with open(data_path, 'rb') as h:
            return pickle.load(h)
    elif chain_type == ChainType.TCR_BETA:
        data_path = os.path.join(module_dir, 'Human_TCRB_HeavyChain_DNA_Orientation_Pipeline.pkl')
        with open(data_path, 'rb') as h:
            return pickle.load(h)
    elif chain_type == ChainType.BCR_LIGHT_KAPPA or chain_type == ChainType.BCR_LIGHT_LAMBDA:
        data_path = os.path.join(module_dir, 'Human_IGH_LightChain_DNA_Orientation_Pipeline.pkl')
        with open(data_path, 'rb') as h:
            return pickle.load(h)
    else:
        raise ValueError(f"Invalid chain type: {chain_type}")