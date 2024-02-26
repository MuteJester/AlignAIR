import subprocess
import pandas as pd
import argparse
import tensorflow as tf
import os
import numpy as np
import random
import tensorflow as tf
import pandas as pd
from Data import HeavyChainDataset, LightChainDataset
from SequenceSimulation.sequence import LightChainSequence
import pickle
from Models.HeavyChain import HeavyChainAlignAIRR
from Models.LightChain import LightChainAlignAIRR
from Trainers import Trainer
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from PostProcessing.HeuristicMatching import HeuristicReferenceMatcher
from PostProcessing.AlleleSelector import DynamicConfidenceThreshold, CappedDynamicConfidenceThreshold
import logging
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


tokenizer_dictionary = {
    "A": 1,
    "T": 2,
    "G": 3,
    "C": 4,
    "N": 5,
    "P": 0,  # pad token
}


def encode_and_equal_pad_sequence(sequence, max_seq_length, tokenizer_dictionary):
    """Encodes a sequence of nucleotides and pads it to the specified maximum length, equally from both sides.

    Args:
        sequence: A sequence of nucleotides.

    Returns:
        A padded sequence, and the start and end indices of the unpadded sequence.
    """

    encoded_sequence = np.array([tokenizer_dictionary[i] for i in sequence])
    padding_length = max_seq_length - len(encoded_sequence)
    iseven = padding_length % 2 == 0
    pad_size = padding_length // 2
    if iseven:
        encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size), 'constant', constant_values=(0, 0))
    else:
        encoded_sequence = np.pad(encoded_sequence, (pad_size, pad_size + 1), 'constant', constant_values=(0, 0))
    return encoded_sequence


def tokenize_chunk(chunk, max_seq_length, tokenizer_dictionary):
    return [(index, encode_and_equal_pad_sequence(sequence, max_seq_length, tokenizer_dictionary)) for index, sequence
            in chunk]


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def tokenize_sequences(sequences, max_seq_length, tokenizer_dictionary, verbose=False):
    num_cpus = cpu_count()
    indexed_sequences = list(enumerate(sequences))
    chunks = chunkify(indexed_sequences, num_cpus)

    # Create a partial function that includes the fixed arguments
    from functools import partial
    tokenize_partial = partial(tokenize_chunk, max_seq_length=max_seq_length, tokenizer_dictionary=tokenizer_dictionary)

    with Pool(num_cpus) as pool:
        if verbose:
            results = list(tqdm(pool.imap(tokenize_partial, chunks), total=len(chunks)))
        else:
            results = pool.map(tokenize_partial, chunks)

    # Flatten the list of lists and sort by the original index to maintain order
    tokenized_sequences = [seq for chunk in results for seq in chunk]
    tokenized_sequences.sort(key=lambda x: x[0])

    # Remove the indices and extract the tokenized sequences
    tokenized_sequences = [seq for index, seq in tokenized_sequences]
    return np.vstack(tokenized_sequences)


def process_csv_and_tokenize(sequences, max_seq_length, tokenizer_dictionary):
    tokenized_matrix = tokenize_sequences(sequences, max_seq_length, tokenizer_dictionary, verbose=True)

    return tokenized_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlingAIRR Model Prediction')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='path to saved alignairr weights')
    parser.add_argument('--save_path', type=str, required=True, help='where to save the outputed predictions')
    parser.add_argument('--chain_type', type=str, required=True, help='heavy / light')
    parser.add_argument('--sequences', type=str, required=True,
                        help='path to csv/tsv file with sequences in a column called "sequence" ')
    parser.add_argument('--lambda_data_config', type=str, required=False, help='path to lambda chain data config')
    parser.add_argument('--kappa_data_config', type=str, required=False, help='path to  kappa chain data config')
    parser.add_argument('--heavy_data_config', type=str, required=False, help='path to heavy chain  data config')

    parser.add_argument('--v_allele_threshold', type=float, default=0.9, help='threshold for v allele prediction')
    parser.add_argument('--d_allele_threshold', type=float, default=0.2, help='threshold for d allele prediction')
    parser.add_argument('--j_allele_threshold', type=float, default=0.8, help='threshold for j allele prediction')
    parser.add_argument('--v_cap', type=int, default=3, help='cap for v allele calls')
    parser.add_argument('--d_cap', type=int, default=3, help='cap for d allele calls')
    parser.add_argument('--j_cap', type=int, default=3, help='cap for j allele calls')

    args = parser.parse_args()
    chain_type = args.chain_type


    ################################### LOAD MODEL #############################################
    logging.info('Loading Dataset Object')
    if chain_type == 'heavy':
        with open(args.heavy_data_config, 'rb') as h:
            heavychain_config = pickle.load(h)

        dataset = HeavyChainDataset(data_path=args.sequences,
                                    dataconfig=heavychain_config, batch_read_file=True)

    elif chain_type == 'light':
        with open(args.kappa_data_config, 'rb') as h:
            lightchain_kappa_config = pickle.load(h)
        with open(args.lambda_data_config, 'rb') as h:
            lightchain_lambda_config = pickle.load(h)

        dataset = LightChainDataset(data_path=args.sequences,
                                    lambda_dataconfig=lightchain_lambda_config,
                                    kappa_dataconfig=lightchain_kappa_config,
                                    batch_read_file=True)
    else:
        raise ValueError(f'Unknown Chain Type: {chain_type}')


    ################################### LOAD DATA #############################################
    logging.info('Reading Target Sequences')
    sep = ',' if '.csv' in args.sequences else '\t'

    try:
        data = pd.read_csv(args.sequences, usecols=['sequence'], sep=sep)
    except ValueError:
        data = pd.read_csv(args.sequences, usecols=['sequence'], sep=sep)

    file_name = args.sequences.split('/')[-1].split('.')[0]

    if args.save_path == 'None':
        args.save_path = args.model_checkpoint.split('saved_models')[0]

    logging.info('Tokenizing Sequences')
    eval_dataset_ = process_csv_and_tokenize(data['sequence'].to_list(), 512, tokenizer_dictionary)
    print('Train Dataset Encoded!', eval_dataset_.shape)

    logging.info('Loading Model')
    print('Init Model..', '> ', args.model_checkpoint)

    trainer = Trainer(
        model=LightChainAlignAIRR if chain_type == 'light' else HeavyChainAlignAIRR,
        dataset=dataset,
        epochs=1,
        steps_per_epoch=1,
        verbose=1,
    )

    trainer.model.build({'tokenized_sequence': (512, 1)})

    MODEL_CHECKPOINT = args.model_checkpoint
    print('Loading: ', MODEL_CHECKPOINT.split('/')[-1])
    trainer.model.load_weights(
        MODEL_CHECKPOINT)
    model = trainer.model
    print('Model Loaded!')

    padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.uint8)
    dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
        'tokenized_sequence': padded_seqs_tensor})
    dataset = (
        dataset_from_tensors
        .batch(512 * 20)
        .prefetch(tf.data.AUTOTUNE)
    )

    ################################### PASS DATA THROUGH MODEL #############################################


    raw_predictions = []

    for i in tqdm(dataset):
        pred = trainer.model.predict(i, verbose=False, batch_size=256)
        raw_predictions.append(pred)

    mutation_rate, v_allele, d_allele, j_allele = [], [], [], []
    v_segment, d_segment, j_segment = [], [], []
    type_ = []

    for i in raw_predictions:
        mutation_rate.append(i['mutation_rate'])
        v_allele.append(i['v_allele'])

        j_allele.append(i['j_allele'])

        v_segment.append(i['v_segment'])

        j_segment.append(i['j_segment'])

        if chain_type == 'light':
            type_.append(i['type'])
        else:
            d_segment.append(i['d_segment'])
            d_allele.append(i['d_allele'])

    mutation_rate = np.vstack(mutation_rate)
    v_allele = np.vstack(v_allele)
    j_allele = np.vstack(j_allele)
    v_segment = np.vstack(v_segment).astype(np.float16)
    j_segment = np.vstack(j_segment).astype(np.float16)
    if chain_type == 'light':
        type_ = np.vstack(type_)
    else:
        d_allele = np.vstack(d_allele)
        d_segment = np.vstack(d_segment).astype(np.float16)

    ################################### POST PROCESS AND SAVE RESULTS #############################################
    # DynamicConfidenceThreshold
    alleles = {'v': v_allele, 'j': j_allele}
    threshold = {'v': args.v_allele_threshold, 'd': args.d_allele_threshold, 'j': args.j_allele_threshold}
    caps = {'v': args.v_cap, 'd': args.d_cap, 'j': args.j_cap}

    if chain_type == 'heavy':
        alleles['d'] = d_allele

    predicted_alleles = {}
    predicted_allele_likelihoods = {}
    threshold_objects = {}
    for _gene in alleles:
        if chain_type == 'heavy':
            extractor = CappedDynamicConfidenceThreshold(heavy_dataconfig=heavychain_config)
        else:
            extractor = CappedDynamicConfidenceThreshold(kappa_dataconfig=lightchain_kappa_config,
                                                         lambda_dataconfig=lightchain_lambda_config)

        threshold_objects[_gene] = extractor
        selected_alleles = extractor.get_alleles(alleles[_gene], confidence=threshold[_gene], n_process=1,
                                                 cap=caps[_gene], allele=_gene)


        predicted_alleles[_gene] = [i[0] for i in selected_alleles]
        predicted_allele_likelihoods[_gene] = [i[1] for i in selected_alleles]
    # HeuristicReferenceMatcher
    segments = {'v': v_segment, 'j': j_segment}
    if chain_type == 'heavy':
        segments['d'] = d_segment

    germline_alignmnets = {}


    for _gene in segments:
        reference_alleles = threshold_objects[_gene].reference_map[_gene]
        mapper = HeuristicReferenceMatcher(reference_alleles)
        mappings = mapper.match(sequences=data['sequence'].to_list(), segments=segments[_gene],
                                alleles=[i[0] for i in predicted_alleles[_gene]])
        germline_alignmnets[_gene] = mappings

    ################################### GENERATE FINAL OUTPUT CSV #############################################

    final_csv = pd.DataFrame({
        'sequence': data['sequence'],
        'v_call': [','.join(i) for i in predicted_alleles['v']],
        'j_call': [','.join(i) for i in predicted_alleles['j']],
        'v_sequence_start': [i['start_in_seq'] for i in germline_alignmnets['v']],
        'v_sequence_end': [i['end_in_seq'] for i in germline_alignmnets['v']],
        'j_sequence_start': [i['start_in_seq'] for i in germline_alignmnets['j']],
        'j_sequence_end': [i['end_in_seq'] for i in germline_alignmnets['j']],
        'v_germline_start': [max(0,i['start_in_ref']) for i in germline_alignmnets['v']],
        'v_germline_end': [i['end_in_ref'] for i in germline_alignmnets['v']],
        'j_germline_start': [max(0,i['start_in_ref']) for i in germline_alignmnets['j']],
        'j_germline_end': [i['end_in_ref'] for i in germline_alignmnets['j']],
        'v_likelihoods': predicted_allele_likelihoods['v'],
        'j_likelihoods': predicted_allele_likelihoods['j']

    })
    if chain_type == 'heavy':
        final_csv['d_sequence_start'] = [i['start_in_seq'] for i in germline_alignmnets['d']]
        final_csv['d_sequence_end'] = [i['end_in_seq'] for i in germline_alignmnets['d']]
        final_csv['d_germline_start'] = [abs(i['start_in_ref']) for i in germline_alignmnets['d']]
        final_csv['d_germline_end'] = [i['end_in_ref'] for i in germline_alignmnets['d']]
        final_csv['d_call'] = [','.join(i) for i in predicted_alleles['d']]
        final_csv['type'] = 'heavy'
    else:
        final_csv['type'] = ['kappa' if i == 1 else 'lambda' for i in type_.astype(int).squeeze()]

    final_csv['mutation_rate'] = mutation_rate

    final_csv.to_csv(args.save_path + file_name + '_alignairr_results.csv', index=False)
