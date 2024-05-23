import argparse
import argparse
import logging
import multiprocessing
import pickle
import time
from multiprocessing import Pool, cpu_count
from multiprocessing import Process
import numpy as np
import pandas as pd
import tensorflow as tf
from GenAIRR.data import builtin_kappa_chain_data_config, builtin_lambda_chain_data_config, \
    builtin_heavy_chain_data_config
from tqdm.auto import tqdm

from src.AlignAIR.Data import LightChainDataset, HeavyChainDataset
from src.AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from src.AlignAIR.Models.LightChain import LightChainAlignAIRR
from src.AlignAIR.PostProcessing.AlleleSelector import CappedDynamicConfidenceThreshold
from src.AlignAIR.Trainers import Trainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(chain_type, config_paths):
    if chain_type == 'heavy':
        if config_paths['heavy'] == 'D':
            return {'heavy': builtin_heavy_chain_data_config()}
        with open(config_paths['heavy'], 'rb') as h:
            return {'heavy': pickle.load(h)}
    elif chain_type == 'light':

        if config_paths['kappa'] == 'D':
            kappa_config = builtin_kappa_chain_data_config()
        else:
            with open(config_paths['kappa'], 'rb') as h:
                kappa_config = pickle.load(h)

        if config_paths['lambda'] == 'D':
            lambda_config = builtin_lambda_chain_data_config()
        else:
            with open(config_paths['lambda'], 'rb') as h:
                lambda_config = pickle.load(h)
        return {'kappa': kappa_config, 'lambda': lambda_config}
    else:
        raise ValueError(f'Unknown Chain Type: {chain_type}')


def read_sequences(file_path):
    sep = ',' if '.csv' in file_path else '\t'
    return pd.read_csv(file_path, usecols=['sequence'], sep=sep)


def sequence_generator(file_path, batch_size=256):
    sep = ',' if '.csv' in file_path else '\t'
    for chunk in pd.read_csv(file_path, usecols=['sequence'], sep=sep, chunksize=batch_size):
        yield chunk['sequence'].tolist()


def tokenize_sequences_batch(sequences, max_seq_length, tokenizer_dictionary):
    tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in
                           sequences]
    return np.vstack(tokenized_sequences)


def load_model(chain_type, model_checkpoint, max_sequence_size, config=None):
    if chain_type == 'heavy':
        dataset = HeavyChainDataset(data_path=args.sequences,
                                                dataconfig=config['heavy'], batch_read_file=True,
                                                max_sequence_length=max_sequence_size)
    elif chain_type == 'light':
        dataset = LightChainDataset(data_path=args.sequences,
                                    lambda_dataconfig=config['lambda'],
                                    kappa_dataconfig=config['kappa'],
                                    batch_read_file=True, max_sequence_length=max_sequence_size)
    else:
        raise ValueError(f'Unknown Chain Type: {chain_type}')

    trainer = Trainer(
        model=LightChainAlignAIRR if chain_type == 'light' else HeavyChainAlignAIRR,
        dataset=dataset,
        epochs=1,
        steps_per_epoch=1,
        verbose=1,
    )

    trainer.model.build({'tokenized_sequence': (max_sequence_size, 1)})

    MODEL_CHECKPOINT = model_checkpoint
    logging.info(f"Loading: {MODEL_CHECKPOINT.split('/')[-1]}")

    trainer.model.load_weights(
        MODEL_CHECKPOINT)
    logging.info(f"Model Loaded Successfully")

    return trainer.model


def make_predictions(model, sequences, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices({'tokenized_sequence': sequences})
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    predictions = []
    for batch in tqdm(dataset):
        predictions.append(model.predict(batch, verbose=0))

    return predictions


def count_rows(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        row_count = sum(1 for row in file)
    return row_count - 1


def sequence_tokenizer_worker(file_path, queue, max_seq_length, tokenizer_dictionary, batch_size=256):
    sep = ',' if '.csv' in file_path else '\t'
    for chunk in pd.read_csv(file_path, usecols=['sequence'], sep=sep, chunksize=batch_size):
        sequences = chunk['sequence'].tolist()
        tokenized_sequences = [encode_and_equal_pad_sequence(seq, max_seq_length, tokenizer_dictionary) for seq in
                               sequences]
        tokenized_batch = np.vstack(tokenized_sequences)
        queue.put(tokenized_batch)
    queue.put(None)


def start_tokenizer_process(file_path, max_seq_length, tokenizer_dictionary, batch_size=256):
    queue = multiprocessing.Queue(maxsize=10)  # Control the prefetching size
    process = Process(target=sequence_tokenizer_worker,
                      args=(file_path, queue, max_seq_length, tokenizer_dictionary, batch_size))
    process.start()
    logging.info('Producer Process Started!')
    return queue, process


def make_predictions(model, sequence_generator, max_sequence_size, total_samples, batch_size=2048):
    predictions = []
    for sequences in tqdm(sequence_generator, total=total_samples // batch_size):
        tokenized_sequences = tokenize_sequences_batch(sequences, max_sequence_size, tokenizer_dictionary)
        # dataset = tf.data.Dataset.from_tensor_slices({'tokenized_sequence': tokenized_sequences})
        # dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # for batch in dataset:
        predictions.append(model.predict({'tokenized_sequence': tokenized_sequences}, verbose=0, batch_size=batch_size))
    return predictions


def post_process_results(predictions, chain_type, config, sequences):
    mutation_rate, v_allele, d_allele, j_allele = [], [], [], []
    v_start, v_end = [], []
    d_start, d_end = [], []
    j_start, j_end = [], []
    indel_count = []
    type_ = []
    productive = []
    for i in predictions:
        mutation_rate.append(i['mutation_rate'])
        v_allele.append(i['v_allele'])
        j_allele.append(i['j_allele'])
        indel_count.append(i['indel_count'])
        productive.append(i['productive'])

        v_start.append(i['v_start'])
        v_end.append(i['v_end'])
        j_start.append(i['j_start'])
        j_end.append(i['j_end'])

        if chain_type == 'light':
            type_.append(i['type'])
        else:
            d_start.append(i['d_start'])
            d_allele.append(i['d_allele'])
            d_end.append(i['d_end'])

    mutation_rate = np.vstack(mutation_rate)
    indel_count = np.vstack(indel_count)
    productive = np.vstack(productive)

    v_allele = np.vstack(v_allele)
    d_allele = np.vstack(d_allele)
    j_allele = np.vstack(j_allele)

    v_start = np.vstack(v_start)
    v_end = np.vstack(v_end)

    j_start = np.vstack(j_start)
    j_end = np.vstack(j_end)

    if chain_type == 'light':
        type_ = np.vstack(type_)
    else:
        d_allele = np.vstack(d_allele)
        d_start = np.vstack(d_start)
        d_end = np.vstack(d_end)

    ################################### POST PROCESS AND SAVE RESULTS #############################################
    # DynamicConfidenceThreshold
    alleles = {'v': v_allele, 'j': j_allele}
    threshold = {'v': args.v_allele_threshold, 'd': args.d_allele_threshold, 'j': args.j_allele_threshold}
    caps = {'v': args.v_cap, 'd': args.d_cap, 'j': args.j_cap}
    segmentation_threshold = {'v': args.v_seg_threshold, 'd': args.d_seg_threshold, 'j': args.j_seg_threshold}

    if chain_type == 'heavy':
        alleles['d'] = d_allele

    predicted_alleles = {}
    predicted_allele_likelihoods = {}
    threshold_objects = {}

    for _gene in alleles:
        if chain_type == 'heavy':
            extractor = CappedDynamicConfidenceThreshold(heavy_dataconfig=config['heavy'])
        else:
            extractor = CappedDynamicConfidenceThreshold(kappa_dataconfig=config['kappa'],
                                                         lambda_dataconfig=config['lambda'])

        threshold_objects[_gene] = extractor
        selected_alleles = extractor.get_alleles(alleles[_gene], confidence=threshold[_gene], n_process=6,
                                                 cap=caps[_gene], allele=_gene)

        predicted_alleles[_gene] = [i[0] for i in selected_alleles]
        predicted_allele_likelihoods[_gene] = [i[1] for i in selected_alleles]

    # HeuristicReferenceMatcher
    # segments = {'v': v_segment, 'j': j_segment}
    # if chain_type == 'heavy':
    #     segments['d'] = d_segment

    # germline_alignmnets = {}

    # for _gene in segments:
    #     reference_alleles = threshold_objects[_gene].reference_map[_gene]
    #     mapper = HeuristicReferenceMatcher(reference_alleles,segment_threshold=segmentation_threshold[_gene])
    #     mappings = mapper.match(sequences=sequences, segments=segments[_gene],
    #                             alleles=[i[0] for i in predicted_alleles[_gene]],threshold=segmentation_threshold[_gene])

    #     germline_alignmnets[_gene] = mappings

    def calculate_pad_size(sequence, max_length=576):
        """
        Calculates the size of padding applied to each side of the sequence
        to achieve the specified maximum length.

        Args:
            sequence_length: The length of the original sequence before padding.
            max_length: The maximum length to which the sequence is padded.

        Returns:
            The size of the padding applied to the start of the sequence.
            If the total padding is odd, one additional unit of padding is applied to the end.
        """

        total_padding = max_length - len(sequence)
        pad_size = total_padding // 2

        return pad_size

    paddings = np.array([calculate_pad_size(i) for i in sequences])
    v_start = np.round((v_start.squeeze() - paddings)).astype(int)
    v_end = np.round((v_end.squeeze() - paddings)).astype(int)

    j_start = np.round((j_start.squeeze() - paddings)).astype(int)
    j_end = np.round((j_end.squeeze() - paddings)).astype(int)

    if chain_type == 'heavy':
        d_start = np.round((d_start.squeeze() - paddings)).astype(int)
        d_end = np.round((d_end.squeeze() - paddings)).astype(int)

    germline_alignmnets = {}
    germline_alignmnets['v'] = [{'start_in_seq': start, 'end_in_seq': end,
                                 'start_in_ref': -1, 'end_in_ref': -1} for start, end in zip(v_start, v_end)
                                ]
    germline_alignmnets['j'] = [{'start_in_seq': start, 'end_in_seq': end,
                                 'start_in_ref': -1, 'end_in_ref': -1} for start, end in zip(j_start, j_end)
                                ]

    if chain_type == 'heavy':
        germline_alignmnets['d'] = [{'start_in_seq': start, 'end_in_seq': end,
                                     'start_in_ref': -1, 'end_in_ref': -1} for start, end in zip(d_start, d_end)
                                    ]

    results = {
        'predicted_alleles': predicted_alleles,
        'germline_alignmnets': germline_alignmnets,
        'predicted_allele_likelihoods': predicted_allele_likelihoods,
        'mutation_rate': mutation_rate,
        'productive': productive,
        'indel_count': indel_count
    }
    if chain_type == 'light':
        results['type_'] = type_
    return results


def save_results(results, save_path, file_name, sequences):
    final_csv = pd.DataFrame({
        'sequence': sequences,
        'v_call': [','.join(i) for i in results['predicted_alleles']['v']],
        'j_call': [','.join(i) for i in results['predicted_alleles']['j']],
        'v_sequence_start': [i['start_in_seq'] for i in results['germline_alignmnets']['v']],
        'v_sequence_end': [i['end_in_seq'] for i in results['germline_alignmnets']['v']],
        'j_sequence_start': [i['start_in_seq'] for i in results['germline_alignmnets']['j']],
        'j_sequence_end': [i['end_in_seq'] for i in results['germline_alignmnets']['j']],
        'v_germline_start': [max(0, i['start_in_ref']) for i in results['germline_alignmnets']['v']],
        'v_germline_end': [i['end_in_ref'] for i in results['germline_alignmnets']['v']],
        'j_germline_start': [max(0, i['start_in_ref']) for i in results['germline_alignmnets']['j']],
        'j_germline_end': [i['end_in_ref'] for i in results['germline_alignmnets']['j']],
        'v_likelihoods': results['predicted_allele_likelihoods']['v'],
        'j_likelihoods': results['predicted_allele_likelihoods']['j'],
    })
    if chain_type == 'heavy':
        final_csv['d_sequence_start'] = [i['start_in_seq'] for i in results['germline_alignmnets']['d']]
        final_csv['d_sequence_end'] = [i['end_in_seq'] for i in results['germline_alignmnets']['d']]
        final_csv['d_germline_start'] = [abs(i['start_in_ref']) for i in results['germline_alignmnets']['d']]
        final_csv['d_germline_end'] = [i['end_in_ref'] for i in results['germline_alignmnets']['d']]
        final_csv['d_call'] = [','.join(i) for i in results['predicted_alleles']['d']]
        final_csv['type'] = 'heavy'
    else:
        final_csv['type'] = ['kappa' if i == 1 else 'lambda' for i in results['type_'].astype(int).squeeze()]

    final_csv['mutation_rate'] = results['mutation_rate']
    final_csv['ar_indels'] = results['indel_count']
    final_csv['ar_productive'] = results['productive']

    final_csv.to_csv(save_path + file_name + '_alignairr_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlingAIR Model Prediction')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='path to saved alignairr weights')
    parser.add_argument('--save_path', type=str, required=True, help='where to save the outputed predictions')
    parser.add_argument('--chain_type', type=str, required=True, help='heavy / light')
    parser.add_argument('--sequences', type=str, required=True,
                        help='path to csv/tsv file with sequences in a column called "sequence" ')
    parser.add_argument('--lambda_data_config', type=str, default='D', help='path to lambda chain data config')
    parser.add_argument('--kappa_data_config', type=str, default='D', help='path to  kappa chain data config')
    parser.add_argument('--heavy_data_config', type=str, default='D', help='path to heavy chain  data config')
    parser.add_argument('--max_input_size', type=int, default=576, help='maximum model input size')
    parser.add_argument('--batch_size', type=int, default=2048, help='The Batch Size for The Model Prediction')

    parser.add_argument('--v_allele_threshold', type=float, default=0.95, help='threshold for v allele prediction')
    parser.add_argument('--d_allele_threshold', type=float, default=0.2, help='threshold for d allele prediction')
    parser.add_argument('--j_allele_threshold', type=float, default=0.8, help='threshold for j allele prediction')
    parser.add_argument('--v_seg_threshold', type=float, default=0.1, help='threshold for v allele segmentation')
    parser.add_argument('--d_seg_threshold', type=float, default=0.01, help='threshold for d allele segmentation')
    parser.add_argument('--j_seg_threshold', type=float, default=0.001, help='threshold for j allele segmentation')
    parser.add_argument('--v_cap', type=int, default=3, help='cap for v allele calls')
    parser.add_argument('--d_cap', type=int, default=3, help='cap for d allele calls')
    parser.add_argument('--j_cap', type=int, default=3, help='cap for j allele calls')

    args = parser.parse_args()
    chain_type = args.chain_type
    tokenizer_dictionary = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}  # pad token

    # Load configuration
    config_paths = {'heavy': args.heavy_data_config, 'kappa': args.kappa_data_config, 'lambda': args.lambda_data_config}
    config = load_config(args.chain_type, config_paths)
    logging.info('Data Config Loaded Successfully')

    # # Read sequences
    # sequences = read_sequences(args.sequences)['sequence'].tolist()
    file_name = args.sequences.split('/')[-1].split('.')[0]
    logging.info(f'Target File : {file_name}')

    # # Tokenize sequences
    # tokenized_sequences = tokenize_sequences(sequences, args.max_input_size, tokenizer_dictionary, verbose=True)

    # # Load model
    # model = load_model(args.chain_type, args.model_checkpoint,args.max_input_size, config)

    # # Make predictions
    # predictions = make_predictions(model, tokenized_sequences,args.max_input_size)

    # Count Rows
    number_of_samples = count_rows(args.sequences)
    logging.info(f'There are : {number_of_samples} Samples for the Model to Predict')
    # Setup the generator
    # sequence_gen = sequence_generator(args.sequences,batch_size=args.batch_size)

    # Load model
    model = load_model(args.chain_type, args.model_checkpoint, args.max_input_size, config)

    # Make predictions
    # predictions = make_predictions(model, sequence_gen, args.max_input_size,total_samples=number_of_samples,batch_size=args.batch_size)
    # Process tokenized batches as they become available
    queue, process = start_tokenizer_process(args.sequences, args.max_input_size, tokenizer_dictionary, args.batch_size)
    predictions = []
    batch_number = 0
    batch_times = []
    start_time = time.time()
    total_batches = int(np.ceil(number_of_samples / args.batch_size))
    while True:
        tokenized_batch = queue.get()
        if tokenized_batch is None:
            break

        batch_start_time = time.time()
        predictions.append(
            model.predict({'tokenized_sequence': tokenized_batch}, verbose=0, batch_size=args.batch_size))
        batch_duration = time.time() - batch_start_time
        batch_times.append(batch_duration)

        batch_number += 1
        avg_batch_time = sum(batch_times) / len(batch_times)
        estimated_time_remaining = avg_batch_time * (total_batches - batch_number)
        time_elapsed = time.time() - start_time

        logging.info(
            f"Processed Batch {batch_number}/{total_batches}. Time Elapsed: {time_elapsed:.2f} seconds. Estimated Time Remaining: {estimated_time_remaining:.2f} seconds.")
    total_duration = time.time() - start_time
    logging.info(f"All batches processed in {total_duration:.2f} seconds.")
    process.join()

    # Post-process results
    sequence_gen = sequence_generator(args.sequences, args.max_input_size)
    sequences = [i for j in sequence_gen for i in j]
    results = post_process_results(predictions, args.chain_type, config, sequences)

    # Save results
    save_results(results, args.save_path, file_name, sequences)
    logging.info(f"Processed Results Saved Successfully at {args.save_path + file_name + '_alignairr_results.csv'}")






