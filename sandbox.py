import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm

from VDeepJDataset import VDeepJDataset

tqdm.pandas()
from VDeepJDataPrepper import VDeepJDataPrepper
from VDeepJModel import VDeepJAllign
from VDeepJUnbondedDataset import VDeepJUnbondedDataset

ub_dataset = VDeepJUnbondedDataset(
    mutation_rate=0.09,
    shm_flat=True,
    randomize_rate=True,
    corrupt_beginning=True,
    corrupt_proba=0.3,
    mutation_oracle_mode=False,
)
