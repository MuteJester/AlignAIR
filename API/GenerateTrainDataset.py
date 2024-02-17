import pandas as pd
from SequenceSimulation.utilities.data_config import DataConfig
from SequenceSimulation.mutation import S5F, Uniform
from SequenceSimulation.simulation import HeavyChainSequenceAugmentor, SequenceAugmentorArguments
from SequenceSimulation.simulation import LightChainKappaLambdaSequenceAugmentor
from SequenceSimulation.simulation import HeavyChainSequenceAugmentor, SequenceAugmentorArguments
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
import threading
import time
import requests
import argparse
from SequenceSimulation.utilities import AlleleNComparer
import pickle
import os

def get_script_path(script_name):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the script
    script_path = os.path.join(current_dir, script_name)

    return script_path


def generate_samples(n, queue):
    print('Process Started')
    print(f'Generating {n} Sequence using {args.mutation_model} Mutation Model')
    print(f'Noise Params > Min: {args.min_mutation_rate}, Max: {args.max_mutation_rate}')
    print(f'V 5 Prime Corrupt Proba: {args.corrupt_proba}')

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
                f' {num_samples} Sequences Are Ready For Use, This Is {1 - ppp}% Left to Be Generated\n')

            buffer = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlingAIRR Sequence Generator')
    parser.add_argument('--dataconfig_path', type=str, required=True, help='data config path')
    parser.add_argument('--mutation_model', type=str, required=True, help='mutation model')
    parser.add_argument('--save_path', type=str, required=True, help='save path')
    parser.add_argument('--n_samples', type=str, required=True, help='number of samples')

    args = parser.parse_args()

    save_path = args.save_path
    BATCH_SIZE = 100_000,
    num_samples = args.n_samples
    MUTATION_MODEL = Uniform if args.mutation_model.lower() == 'uniform' else S5F
    CMMP = None,
    DATACONFIG = args.dataconfig_path

    with open(DATACONFIG, 'rb') as h:
        dc = pickle.load(h)
    print('Using DataConfig: ', DATACONFIG)

    num_cores = cpu_count()
    samples_per_core = int(num_samples) // num_cores

    # Create a shared queue
    manager = Manager()
    queue = manager.Queue()

    # Start the writer thread
    writer = threading.Thread(target=writer_thread, args=(queue,))
    writer.start()
    # load dataconfig
    import pickle

    with open(DATACONFIG, 'rb') as h:
        heavychain_config = pickle.load(h)

    # Create initial CSV with headers
    args = SequenceAugmentorArguments(mutation_model=MUTATION_MODEL, custom_mutation_model_path=CMMP)
    simulator = HeavyChainSequenceAugmentor(heavychain_config, args)

    df = pd.DataFrame(columns=simulator.columns)
    df.to_csv(save_path, index=False)

    # Start the processes
    with Pool(num_cores) as pool:
        pool.starmap(generate_samples, [(samples_per_core, queue) for _ in range(num_cores)])

    # Signal the writer thread to stop
    queue.put("STOP")
    writer.join()
