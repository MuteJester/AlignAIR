import sys

# Let's say your module is in '/path/to/your/module'
module_dir = '/home/bcrlab/thomas/AlignAIRR/'

# Append this directory to sys.path
if module_dir not in sys.path:
    sys.path.append(module_dir)
    
import pandas as pd
from SequenceSimulation.utilities.data_config import DataConfig
from SequenceSimulation.mutation import S5F,Uniform
from SequenceSimulation.simulation import HeavyChainSequenceAugmentor,SequenceAugmentorArguments
from SequenceSimulation.simulation import LightChainKappaLambdaSequenceAugmentor
from SequenceSimulation.simulation import HeavyChainSequenceAugmentor,SequenceAugmentorArguments
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
import threading
import time
import requests
import argparse
from SequenceSimulation.utilities import AlleleNComparer

import os

def get_script_path(script_name):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the script
    script_path = os.path.join(current_dir, script_name)

    return script_path

parameters = {
    
    'Uniform':dict(
        save_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/HeavyChain_OGRDB_DataConfig_AlignAIRR_Uniform_15M_with_Corruption_Mrate_003__025.csv",
        BATCH_SIZE = 100_000,
        num_samples = 15_000_000,
        MUTATION_MODEL = Uniform,
        CMMP = None,
        DATACONFIG = './SequenceSimulation/data/HeavyChain_DataConfig_OGRDB.pkl'
        ),
    'S5F':dict(
        save_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/HeavyChain_OGRDB_DataConfig_AlignAIRR_S5F_15M_with_Corruption_Mrate_003__025.csv",
        BATCH_SIZE = 100_000,
        num_samples = 15_000_000,
        MUTATION_MODEL = S5F,
        CMMP = None,
        DATACONFIG = './SequenceSimulation/data/HeavyChain_DataConfig_OGRDB.pkl'
        ),
    'S5F_60':dict(
        save_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/HeavyChain_OGRDB_DataConfig_AlignAIRR_S5F_60_15M_with_Corruption_Mrate_003__025.csv",
        BATCH_SIZE = 100_000,
        num_samples = 15_000_000,
        MUTATION_MODEL = S5F,
        CMMP = './SequenceSimulation/data/HH_S5F_60_META.pkl',
        DATACONFIG = './SequenceSimulation/data/HeavyChain_DataConfig_OGRDB.pkl'
        ),
    'S5F_Opposite':dict(
        save_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/HeavyChain_OGRDB_DataConfig_AlignAIRR_S5F_Opposite_15M_with_Corruption_Mrate_003__025.csv",
        BATCH_SIZE = 100_000,
        num_samples = 15_000_000,
        MUTATION_MODEL = S5F,
        CMMP = './SequenceSimulation/data/HH_S5F_Opposite_META.pkl',
        DATACONFIG = './SequenceSimulation/data/HeavyChain_DataConfig_OGRDB.pkl'
        )
}


# model = 'S5F'
import os
model = os.getenv('MODEL', 'S5F')  # Default to 'S5F' if MODEL env variable is not set

save_path = parameters[model]['save_path']
BATCH_SIZE = parameters[model]['BATCH_SIZE']
num_samples = parameters[model]['num_samples']
MUTATION_MODEL = parameters[model]['MUTATION_MODEL']
CMMP = parameters[model]['CMMP']
DATACONFIG = parameters[model]['DATACONFIG']

print(CMMP)


with open(DATACONFIG,'rb') as h:
    import pickle
    dc = pickle.load(h)
print('Using DataConfig: ',DATACONFIG)

def generate_samples(n, queue):
    print('Process Started')
    print(f'Generating {n} Sequence using {args.mutation_model} Mutation Model')
    print(f'Noise Params > Min: {args.min_mutation_rate}, Max: {args.max_mutation_rate}')
    print(f'Corrupt Proba: {args.corrupt_proba}')
    
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
                ppp = np.round(written_count/num_samples,3)
            break
        buffer.append(item)
        if len(buffer) >= BATCH_SIZE:
            df = pd.DataFrame(buffer)
            df.to_csv(save_path, mode='a', header=False, index=False)
            written_count += BATCH_SIZE
            ppp = np.round(written_count/num_samples,3)
            print(f'Hey Thomas!\n{written_count} Sequences Are Ready For Use, This Is {1-ppp}% Left to Be Generated ({model} Model!) \n')

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
    import pickle
    with open(DATACONFIG,'rb') as h:
        heavychain_config = pickle.load(h)

    # Create initial CSV with headers
    args = SequenceAugmentorArguments(mutation_model=MUTATION_MODEL,custom_mutation_model_path=CMMP)
    simulator = HeavyChainSequenceAugmentor(heavychain_config,args)

    df = pd.DataFrame(columns=simulator.columns)
    df.to_csv(save_path, index=False)

   
        
    # Start the processes
    with Pool(num_cores) as pool:
        pool.starmap(generate_samples, [(samples_per_core, queue) for _ in range(num_cores)])

    # Signal the writer thread to stop
    queue.put("STOP")
    writer.join()


















