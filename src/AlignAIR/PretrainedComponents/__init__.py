import pickle
import os

module_dir = os.path.dirname(__file__)

def builtin_orientation_classifier():
    data_path = os.path.join(module_dir, 'DNA_Orientation_Pipeline.pkl')
    with open(data_path, 'rb') as h:
        return pickle.load(h)