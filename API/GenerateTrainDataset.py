import pandas as pd
from ..SequenceSimulation.mutation import S5F, Uniform
from ..SequenceSimulation.simulation import HeavyChainSequenceAugmentor, SequenceAugmentorArguments,LightChainKappaLambdaSequenceAugmentor
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
import threading
import os
import argparse
import pickle


def generate_samples(n, queue,heavychain_config, args):
    print('Process Started')
    print(f'Generating {n} Sequences using {args.mutation_model} Mutation Model')
    if type(heavychain_config) != tuple:
        simulator = HeavyChainSequenceAugmentor(heavychain_config, args)
    else:
        simulator = LightChainKappaLambdaSequenceAugmentor(lambda_dataconfig=heavychain_config[0],
                          kappa_dataconfig=heavychain_config[1], lambda_args=args,kappa_args=args)

    for _ in range(n):
        simulation = simulator.simulate_augmented_sequence()
        queue.put(simulation)

def writer_thread(queue, save_path, batch_size, num_samples):
    buffer = []
    written_count = 0
    while True:
        item = queue.get()
        if item == "STOP":
            break
        buffer.append(item)
        if len(buffer) >= batch_size:
            df = pd.DataFrame(buffer)
            df.to_csv(save_path, mode='a', header=False, index=False)
            written_count += len(buffer)
            buffer = []
    if buffer:  # Write any remaining items in buffer
        df = pd.DataFrame(buffer)
        df.to_csv(save_path, mode='a', header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlingAIRR Sequence Generator')
    parser.add_argument('--dataconfig_path', type=str, required=True, help='Data config path for heavy chain or lambda chain')
    parser.add_argument('--dataconfig_kappa', type=str, required=False, help='Data config path for kappa chain')
    parser.add_argument('--mutation_model', type=str, required=True, help='Mutation model')
    parser.add_argument('--save_path', type=str, required=True, help='Save path')
    parser.add_argument('--n_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('--chain_type', type=str, required=True, choices=['light', 'heavy'], help='Chain type: light or heavy')

    # New parameters
    parser.add_argument('--min_mutation_rate', type=float, required=True, help='Minimum mutation rate')
    parser.add_argument('--max_mutation_rate', type=float, required=True, help='Maximum mutation rate')
    parser.add_argument('--n_ratio', type=float, required=True, help="Ratio of N's inserted to the sequence")
    parser.add_argument('--max_sequence_length', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--nucleotide_add_coefficient', type=float, required=True, help='Coefficient for the nucleotide add distribution')
    parser.add_argument('--nucleotide_remove_coefficient', type=float, required=True, help='Coefficient for the nucleotide remove distribution')
    parser.add_argument('--nucleotide_add_after_remove_coefficient', type=float, required=True, help='Coefficient for the nucleotide add after remove distribution')
    parser.add_argument('--random_sequence_add_proba', type=float, required=True, help='Probability of adding a random sequence')
    parser.add_argument('--single_base_stream_proba', type=float, required=True, help='Probability of adding a single base stream')
    parser.add_argument('--duplicate_leading_proba', type=float, required=True, help='Probability of duplicating the leading base')
    parser.add_argument('--random_allele_proba', type=float, required=True, help='Probability of adding a random allele')
    parser.add_argument('--corrupt_proba', type=float, required=True, help='Probability of corrupting the sequence from the start')
    parser.add_argument('--short_d_length', type=int, required=True, help='Minimum length required from the D allele to not be tagged as "Short-D"')
    parser.add_argument('--save_mutations_record', type=int, help='Whether to save the mutations in the sequence')
    parser.add_argument('--save_ns_record', type=int, help="Whether to save the N's in the sequence")

    args = parser.parse_args()

    BATCH_SIZE = 100000
    MUTATION_MODEL = Uniform if args.mutation_model.lower() == 'uniform' else S5F


    if args.chain_type.lower() == 'heavy':
        with open(args.dataconfig_path, 'rb') as h:
            heavychain_config = pickle.load(h)
        print('Using DataConfig: ', args.dataconfig_path)
    else:
        with open(args.dataconfig_path, 'rb') as h:
            lambda_config = pickle.load(h)
        print('Using DataConfig: ', args.dataconfig_path)
        if args.dataconfig_kappa is None:
            raise ValueError('No Kappa Config Provided')
        with open(args.dataconfig_kappa, 'rb') as h:
            kappa_config = pickle.load(h)
        print('Using DataConfig: ', args.dataconfig_path)

    sequence_args = SequenceAugmentorArguments(
        mutation_model=MUTATION_MODEL,
        min_mutation_rate=args.min_mutation_rate,
        max_mutation_rate=args.max_mutation_rate,
        n_ratio=args.n_ratio,
        max_sequence_length=args.max_sequence_length,
        nucleotide_add_coefficient=args.nucleotide_add_coefficient,
        nucleotide_remove_coefficient=args.nucleotide_remove_coefficient,
        nucleotide_add_after_remove_coefficient=args.nucleotide_add_after_remove_coefficient,
        random_sequence_add_proba=args.random_sequence_add_proba,
        single_base_stream_proba=args.single_base_stream_proba,
        duplicate_leading_proba=args.duplicate_leading_proba,
        random_allele_proba=args.random_allele_proba,
        corrupt_proba=args.corrupt_proba,
        short_d_length=args.short_d_length,
        save_mutations_record=bool(int(args.save_mutations_record)),
        save_ns_record=bool(int(args.save_ns_record))
    )

    manager = Manager()
    queue = manager.Queue()

    writer = threading.Thread(target=writer_thread, args=(queue, args.save_path, BATCH_SIZE, args.n_samples))
    writer.start()

    if args.chain_type == 'heavy':
        df = pd.DataFrame(columns=HeavyChainSequenceAugmentor(heavychain_config, sequence_args).columns)
        df.to_csv(args.save_path, index=False)
    else:
        df = pd.DataFrame(columns=LightChainKappaLambdaSequenceAugmentor(lambda_dataconfig=lambda_config,
                          kappa_dataconfig=kappa_config, lambda_args=sequence_args,kappa_args=sequence_args).columns)
        df.to_csv(args.save_path, index=False)

    num_cores = cpu_count()
    samples_per_core = args.n_samples // num_cores

    if args.chain_type == 'heavy':
        with Pool(num_cores) as pool:
            pool.starmap(generate_samples, [(samples_per_core, queue,heavychain_config, sequence_args) for _ in range(num_cores)])
    else:
        with Pool(num_cores) as pool:
            pool.starmap(generate_samples,
                         [(samples_per_core, queue, (lambda_config,kappa_config), sequence_args) for _ in range(num_cores)])

    queue.put("STOP")
    writer.join()
