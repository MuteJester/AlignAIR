import random
from itertools import product
from GenAIRR.data import builtin_heavy_chain_data_config
from GenAIRR.mutation import S5F
from GenAIRR.simulation import SequenceAugmentorArguments, HeavyChainSequenceAugmentor
from tqdm.auto import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
import logging

from GenAIRR.alleles.allele import AlleleTypes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def simulate_sequence(agumentor, V, D, J, tol=10):
    # logging.info(f"Simulating sequence for V: {V.name}, D: {D.name if type(D) != str else D}, J: {J.name}")
    if D != 'Short-D':
        gen = agumentor.simulate_augmented_sequence(specific_v=V, specific_d=D, specific_j=J)
        t = 0

        while 'Short-D' in gen['d_call'] and t < tol:
            # logging.info(f"Retrying simulation due to 'Short-D' for V: {V.name}, D: {D.name if type(D) != str else D}, J: {J.name}, attempt {t + 1}")
            gen = agumentor.simulate_augmented_sequence(specific_v=V, specific_d=D, specific_j=J)
            t += 1

        # logging.info(f"Completed simulation for V: {V.name}, D: {D.name if type(D) != str else D}, J: {J.name}")
        return gen
    else:
        gen = agumentor.simulate_augmented_sequence(specific_v=V, specific_j=J)
        while 'Short-D' not in gen['d_call']:
            gen = agumentor.simulate_augmented_sequence(specific_v=V, specific_j=J)

        # logging.info(f"Completed simulation for V: {V.name}, D: {D.name if type(D) != str else D}, J: {J.name}")
        return gen


def process_combinations(args):
    agumentor, combinations, _alleles = args
    dataset = []
    for combination in combinations:
        if len(combination) == 2:
            if type(combination[0]) == str or combination[0].type == AlleleTypes.D:
                D, J = combination
                _v, _d, _j = random.choice(_alleles), D, J
            else:
                V, D = combination
                _v, _d, _j = V, D, random.choice(_alleles)
        else:
            V, D, J = combination
            _v, _d, _j = V, D, J
        gen = simulate_sequence(agumentor, _v, _d, _j)
        dataset.append(gen)
    return dataset


def create_uniform_balanced_dataset(agumentor, K=2):
    logging.info("Starting dataset creation process")

    # Generate allele lists
    logging.info("Generating allele lists")
    V_alleles = [i for j in agumentor.dataconfig.v_alleles for i in agumentor.dataconfig.v_alleles[j]]
    D_alleles = [i for j in agumentor.dataconfig.d_alleles for i in agumentor.dataconfig.d_alleles[j]]
    J_alleles = [i for j in agumentor.dataconfig.j_alleles for i in agumentor.dataconfig.j_alleles[j]]

    V_alleles = [i for i in V_alleles if i.name != 'IGHVF9-G32*02']
    D_alleles.append("Short-D")

    # Repeat each allele K times
    logging.info("Repeating alleles K times")
    V_alleles = V_alleles * K
    D_alleles = D_alleles * K
    J_alleles = J_alleles * K

    # Shuffle each list
    logging.info("Shuffling allele lists")
    random.shuffle(V_alleles)
    random.shuffle(D_alleles)
    random.shuffle(J_alleles)

    # Generate uniform combinations
    logging.info("Generating uniform combinations")
    V_D_combinations = list(product(V_alleles, D_alleles))
    D_J_combinations = list(product(D_alleles, J_alleles))
    V_D_J_combinations = list(product(V_alleles, D_alleles, J_alleles))

    total_combinations = len(V_D_combinations) + len(D_J_combinations) + len(V_D_J_combinations)
    logging.info(f"Total combinations to process: {total_combinations}")

    # Prepare arguments for multiprocessing
    num_cores = cpu_count()
    chunk_size_VD = len(V_D_combinations) // num_cores
    chunk_size_DJ = len(D_J_combinations) // num_cores
    chunk_size_VDJ = len(V_D_J_combinations) // num_cores

    args_list = [(agumentor, V_D_combinations[i:i + chunk_size_VD], J_alleles) for i in
                 range(0, len(V_D_combinations), chunk_size_VD)]

    args_list += [(agumentor, D_J_combinations[i:i + chunk_size_DJ], V_alleles) for i in
                  range(0, len(D_J_combinations), chunk_size_DJ)]

    args_list += [(agumentor, V_D_J_combinations[i:i + chunk_size_VDJ], None) for i in
                  range(0, len(V_D_J_combinations), chunk_size_VDJ)]

    # Use multiprocessing Pool
    logging.info(f"Starting multiprocessing pool with {num_cores} cores")
    with Pool(num_cores) as pool:
        results = list(
            tqdm(pool.imap(process_combinations, args_list), total=len(args_list), desc='Generating dataset'))

    # Combine results
    logging.info("Combining results")
    dataset = [item for sublist in results for item in sublist]

    logging.info("Dataset creation process completed")
    return pd.DataFrame(dataset)


if __name__ == '__main__':
    logging.info("Script started")

    heavychain_config = builtin_heavy_chain_data_config()
    CCMP = None
    MUTATION_MODEL = S5F
    productive = False

    if productive:
        args = SequenceAugmentorArguments(mutation_model=MUTATION_MODEL, custom_mutation_model_path=CCMP,
                                          simulate_indels=0,
                                          corrupt_proba=0, productive=True, max_sequence_length=560)
    else:
        args = SequenceAugmentorArguments(mutation_model=MUTATION_MODEL, custom_mutation_model_path=CCMP,
                                          simulate_indels=0.2,
                                          corrupt_proba=0.7, productive=False, max_sequence_length=560)

    simulator = HeavyChainSequenceAugmentor(args=args, dataconfig=heavychain_config)

    K = 2  # Number of samples for each allele

    file_name = 'S5F_Train_Dataset_'
    file_name += 'Productive' if productive else 'Non_Productive'
    file_name += '.csv'

    logging.info("Starting dataset generation")
    balanced_dataset = create_uniform_balanced_dataset(simulator, K)
    save_path = '/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/'
    balanced_dataset.to_csv(f"{save_path}{file_name}", index=False)
    logging.info(f"Dataset saved to {save_path}{file_name}")
    logging.info("Script finished")
