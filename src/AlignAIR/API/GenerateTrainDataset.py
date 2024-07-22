import sys

# Let's say your module is in '/path/to/your/module'
module_dir = '/home/bcrlab/thomas/AlignAIRR/'

# Append this directory to sys.path
if module_dir not in sys.path:
    sys.path.append(module_dir)

import pandas as pd
from GenAIRR.utilities.data_config import DataConfig
from GenAIRR.mutation import S5F, Uniform
from GenAIRR.simulation import HeavyChainSequenceAugmentor, SequenceAugmentorArguments
from GenAIRR.simulation import LightChainKappaLambdaSequenceAugmentor
from GenAIRR.simulation import HeavyChainSequenceAugmentor, SequenceAugmentorArguments
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
import threading
import time
import requests
import argparse
from GenAIRR.utilities import AlleleNComparer

parameters = {

    'Uniform': dict(
        save_path="/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/HeavyChain_OGRDBV8_DataConfig_AlignAIRR_S5F_10M_with_Corruption_Mrate_003__025_Mixed.csv",
        BATCH_SIZE=200_000,
        num_samples=10_000_000,
        MUTATION_MODEL=S5F,
        CMMP=None,
    ),
    'S5F': dict(
        save_path="/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/HeavyChain_OGRDBV8_DataConfig_AlignAIRR_S5F_15M_with_Corruption_Mrate_003__025_Productive.csv",
        BATCH_SIZE=200_000,
        num_samples=15_000_000,
        MUTATION_MODEL=S5F,
        CMMP=None,
    ),
    'S5F_60': dict(
        save_path="/localdata/alignairr_data/AlignAIRR_Evaluation_Dataset/HeavyChain_OGRDBV7_DataConfig_AlignAIRR_S5F_60_3M_with_Corruption_Mrate_003__025_Mixed.csv",
        BATCH_SIZE=100_000,
        num_samples=3_000_000,
        MUTATION_MODEL=S5F,
        CMMP='/home/bcrlab/thomas/AlignAIRR/SequenceSimulation/data/HH_S5F_60_META.pkl',
    ),
    'S5F_Opposite': dict(
        save_path="/localdata/alignairr_data/AlignAIRR_Evaluation_Dataset/HeavyChain_OGRDBV7_DataConfig_AlignAIRR_S5F_Opposite_3M_with_Corruption_Mrate_003__025_Mixed.csv",
        BATCH_SIZE=100_000,
        num_samples=3_000_000,
        MUTATION_MODEL=S5F,
        CMMP='/home/bcrlab/thomas/AlignAIRR/SequenceSimulation/data/HH_S5F_Opposite_META.pkl',
    )
}

import os

model = os.getenv('MODEL', 'S5F')  # Default to 'S5F' if MODEL env variable is not set

save_path = parameters[model]['save_path']
BATCH_SIZE = parameters[model]['BATCH_SIZE']
num_samples = parameters[model]['num_samples']
MUTATION_MODEL = parameters[model]['MUTATION_MODEL']
CMMP = parameters[model]['CMMP']


def generate_samples(n, queue):
    print('Process Started')
    print(f'Generating {n} Sequence using {args.mutation_model} Mutation Model')
    print(f'Noise Params > Min: {args.min_mutation_rate}, Max: {args.max_mutation_rate}')
    print(f'Corrupt Proba: {args.corrupt_proba}')
    print('Productive Only:', args.productive)
    for _ in range(n):
        simulation = simulator.simulate_augmented_sequence()
        queue.put(simulation)


def writer_thread(queue):
    buffer = []
    written_count = 0  # Keep track of the number of sequences written to the file

    while True:
        item = queue.get()
        if item == "STOP":
            if buffer:
                df = pd.DataFrame(buffer)
                df.to_csv(save_path, mode='a', header=False, index=False)
                written_count += len(buffer)
            if written_count % 100_000:
                ppp = np.round(written_count / num_samples, 3)
            break
        buffer.append(item)
        if len(buffer) >= BATCH_SIZE:
            df = pd.DataFrame(buffer)
            df.to_csv(save_path, mode='a', header=False, index=False)
            written_count += BATCH_SIZE
            ppp = np.round(written_count / num_samples, 3)
            print(
                f'Hey Thomas!\n{written_count} Sequences Are Ready For Use, This Is {1 - ppp}% Left to Be Generated ({model} Model!) \n')

            buffer = []


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Evaluation Dataset Simulation Script')
    # parser.add_argument('--model', type=str, required=True, help='mutation model')

    # args = parser.parse_args()

    num_cores = cpu_count()
    samples_per_core = num_samples // num_cores

    # Create a shared queue
    manager = Manager()
    queue = manager.Queue()

    # Start the writer thread
    writer = threading.Thread(target=writer_thread, args=(queue,))
    writer.start()
    # load dataconfig
    # import pickle
    # with open(DATACONFIG,'rb') as h:
    #     heavychain_config = pickle.load(h)
    from GenAIRR.data import builtin_heavy_chain_data_config, builtin_kappa_chain_data_config, \
        builtin_lambda_chain_data_config

    heavychain_config = builtin_heavy_chain_data_config()
    kappa_config = builtin_kappa_chain_data_config()
    lambda_config = builtin_lambda_chain_data_config()

    # Create initial CSV with headers
    args = SequenceAugmentorArguments(mutation_model=MUTATION_MODEL, custom_mutation_model_path=CMMP, simulate_indels=0,
                                      corrupt_proba=0, productive=True)

    # simulator = LightChainKappaLambdaSequenceAugmentor(kappa_dataconfig=kappa_config,lambda_dataconfig=lambda_config,lambda_args=args,kappa_args=args)
    simulator = HeavyChainSequenceAugmentor(args=args, dataconfig=heavychain_config)
    df = pd.DataFrame(columns=simulator.columns)
    print(simulator.columns)
    df.to_csv(save_path, index=False)

    # Start the processes
    with Pool(num_cores) as pool:
        pool.starmap(generate_samples, [(samples_per_core, queue) for _ in range(num_cores)])

    # Signal the writer thread to stop
    queue.put("STOP")
    writer.join()


















