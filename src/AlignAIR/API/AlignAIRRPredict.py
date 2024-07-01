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

from AlignAIR.Data import LightChainDataset, HeavyChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR
from AlignAIR.PostProcessing.AlleleSelector import CappedDynamicConfidenceThreshold
from AlignAIR.PostProcessing.HeuristicMatching import HeuristicReferenceMatcher
from AlignAIR.PretrainedComponents import builtin_orientation_classifier
from AlignAIR.Trainers import Trainer
from AlignAIR.Utilities.consumer_producer import sequence_tokenizer_worker, READER_WORKER_TYPES
from AlignAIR.Utilities.file_processing import count_rows, tabular_sequence_generator, FILE_SEQUENCE_GENERATOR, \
    FILE_ROW_COUNTERS
from AlignAIR.Utilities.sequence_processing import tokenize_sequences_batch
from AlignAIR.PostProcessing.AlleleNameTranslation import TranslateToIMGT
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tokenizer_dictionary = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5, "P": 0}  # pad token

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

def load_model(sequences,chain_type, model_checkpoint, max_sequence_size, config=None):
    if chain_type == 'heavy':
        dataset = HeavyChainDataset(data_path=sequences,
                                                dataconfig=config['heavy'], batch_read_file=True,
                                                max_sequence_length=max_sequence_size)
    elif chain_type == 'light':
        dataset = LightChainDataset(data_path=sequences,
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

    trainer.model.load_weights(MODEL_CHECKPOINT)

    logging.info(f"Loading: {MODEL_CHECKPOINT.split('/')[-1]}")

    logging.info(f"Model Loaded Successfully")

    return trainer.model

def start_tokenizer_process(file_path, max_seq_length, tokenizer_dictionary,logger,orientation_pipeline, batch_size=256):
    queue = multiprocessing.Queue(maxsize=64)  # Control the prefetching size
    file_type = file_path.split('.')[-1] # get the file type i.e .csv,.tsv or .fasta
    worker_reading_type = READER_WORKER_TYPES[file_type]
    process = Process(target=worker_reading_type,
                      args=(file_path, queue, max_seq_length, tokenizer_dictionary, batch_size,logger,orientation_pipeline))
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

def clean_and_arrange_predictions(predictions,chain_type):
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
    productive = np.vstack(productive) > 0.5

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
        d_start = np.vstack(d_start)
        d_end = np.vstack(d_end)
        d_allele = np.vstack(d_allele)

    return (v_allele,d_allele,j_allele,v_start,v_end,d_start,d_end,
            j_start,j_end,mutation_rate,indel_count,productive,type_)

def correct_segments_for_paddings(sequences,chain_type,v_start,v_end,d_start,d_end,j_start,j_end):
    paddings = np.array([calculate_pad_size(i) for i in sequences])

    v_start = np.round((v_start.squeeze() - paddings)).astype(int)
    v_end = np.round((v_end.squeeze() - paddings)).astype(int)

    j_start = np.round((j_start.squeeze() - paddings)).astype(int)
    j_end = np.round((j_end.squeeze() - paddings)).astype(int)

    if chain_type == 'heavy':
        d_start = np.round(np.vstack(d_start).squeeze()).astype(int)
        d_end = np.round(np.vstack(d_end).squeeze()).astype(int)

    return v_start,v_end,d_start,d_end,j_start,j_end

def extract_likelihoods_and_labels_from_calls(args,alleles,threshold,caps,config,chain_type):
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
        selected_alleles = extractor.get_alleles(alleles[_gene], confidence=threshold[_gene],
                                                 cap=caps[_gene], allele=_gene)


        predicted_alleles[_gene] = [i[0] for i in selected_alleles]


        predicted_allele_likelihoods[_gene] = [i[1] for i in selected_alleles]

    return predicted_alleles,predicted_allele_likelihoods,threshold_objects


def align_with_germline(segments,threshold_objects,predicted_alleles,sequences):
    germline_alignmnets = {}

    for _gene in segments:
        reference_alleles = threshold_objects[_gene].reference_map[_gene]
        if _gene == 'd':
            reference_alleles['Short-D'] = ''

        starts,ends = segments[_gene]
        mapper = HeuristicReferenceMatcher(reference_alleles)
        mappings = mapper.match(sequences=sequences, starts=starts,ends=ends,
                                alleles=[i[0] for i in predicted_alleles[_gene]],_gene=_gene)

        germline_alignmnets[_gene] = mappings

    return germline_alignmnets

def post_process_results(args,predictions, chain_type, config, sequences):


    (v_allele,d_allele,j_allele,
     v_start,v_end,
     d_start,d_end,
     j_start,j_end,
     mutation_rate,indel_count,productive,type_) = clean_and_arrange_predictions(predictions,args.chain_type)

    v_start, v_end, d_start, d_end, j_start, j_end = correct_segments_for_paddings(sequences,args.chain_type,v_start,
                                                                                   v_end,d_start,d_end,j_start,j_end)

    # DynamicConfidenceThreshold
    alleles = {'v': v_allele, 'j': j_allele}
    threshold = {'v': args.v_allele_threshold, 'd': args.d_allele_threshold, 'j': args.j_allele_threshold}
    caps = {'v': args.v_cap, 'd': args.d_cap, 'j': args.j_cap}

    if chain_type == 'heavy':
        alleles['d'] = d_allele

    predicted_alleles,predicted_allele_likelihoods,threshold_objects = (
        extract_likelihoods_and_labels_from_calls(args,alleles,threshold,caps,config,args.chain_type))


    segments = {'v': [v_start,v_end], 'j': [j_start,j_end]}
    if chain_type == 'heavy':
        segments['d'] = [d_start,d_end]

    germline_alignmnets = align_with_germline(segments, threshold_objects, predicted_alleles, sequences)

    if not args.translate_to_asc:
        translator = TranslateToIMGT(config)
        predicted_alleles['v'] = [[translator.translate(j) for j in i] for i in predicted_alleles['v']]

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


def save_results(results, save_path, file_name, sequences,chain_type):

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='AlingAIR Model Prediction')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='path to saved alignair weights')
    parser.add_argument('--save_path', type=str, required=True, help='where to save the alignment')
    parser.add_argument('--chain_type', type=str, required=True, help='heavy / light')
    parser.add_argument('--sequences', type=str, required=True,
                        help='path to csv/tsv file with sequences in a column called "sequence" ')
    parser.add_argument('--lambda_data_config', type=str, default='D', help='path to lambda chain data config')
    parser.add_argument('--kappa_data_config', type=str, default='D', help='path to  kappa chain data config')
    parser.add_argument('--heavy_data_config', type=str, default='D', help='path to heavy chain  data config')
    parser.add_argument('--max_input_size', type=int, default=576, help='maximum model input size, NOTE! this is with respect to the dimensions the model was trained on, do not increase for pretrained models')
    parser.add_argument('--batch_size', type=int, default=2048, help='The Batch Size for The Model Prediction')

    parser.add_argument('--v_allele_threshold', type=float, default=0.95, help='threshold for v allele prediction')
    parser.add_argument('--d_allele_threshold', type=float, default=0.2, help='threshold for d allele prediction')
    parser.add_argument('--j_allele_threshold', type=float, default=0.8, help='threshold for j allele prediction')
    parser.add_argument('--v_cap', type=int, default=3, help='cap for v allele calls')
    parser.add_argument('--d_cap', type=int, default=3, help='cap for d allele calls')
    parser.add_argument('--j_cap', type=int, default=3, help='cap for j allele calls')

    # For Post Processing
    parser.add_argument('--translate_to_asc', action='store_true',
                        help='Translate names back to ASCs names from IMGT')

    # For Pre Processing
    parser.add_argument('--fix_orientation', type=bool,default=True,
                        help='Adds a preprocessing steps that tests and fixes the DNA orientation, in case it is '
                             'reversed,compliment or reversed and compliment')
    parser.add_argument('--custom_orientation_pipeline_path', type=str, default=None,
                        help='a path to a custom orientation model created for a custom reference')

    args = parser.parse_args()
    return args
def main():
    args = parse_arguments()
    chain_type = args.chain_type

    # Load configuration
    config_paths = {'heavy': args.heavy_data_config,
                    'kappa': args.kappa_data_config,
                    'lambda': args.lambda_data_config
                    }

    config = load_config(args.chain_type, config_paths)
    logging.info('Data Config Loaded Successfully')

    # # Read sequences
    file_name = args.sequences.split('/')[-1].split('.')[0]
    file_type = args.sequences.split('.')[-1]
    logging.info(f'Target File : {file_name}')

    # Count Rows
    row_counter = FILE_ROW_COUNTERS[file_type]
    number_of_samples = row_counter(args.sequences)
    logging.info(f'There are : {number_of_samples} Samples for the Model to Predict')

    # Load model
    model = load_model(args.sequences,args.chain_type, args.model_checkpoint, args.max_input_size, config)

    # Load DNA orientation model
    orientation_pipeline = None
    if args.fix_orientation:
        if args.custom_orientation_pipeline_path is not None:
            with open(args.custom_orientation_pipeline_path,'rb') as h:
                orientation_pipeline = pickle.load(h)
        else:
            orientation_pipeline = builtin_orientation_classifier()
    logging.info('Orientation Pipeline Loaded Successfully')



    # Process tokenized batches as they become available
    queue, process = start_tokenizer_process(args.sequences, args.max_input_size, tokenizer_dictionary,orientation_pipeline,
                                             args.batch_size,)
    predictions = []
    sequences = []
    batch_number = 0
    batch_times = []
    start_time = time.time()
    total_batches = int(np.ceil(number_of_samples / args.batch_size))
    while True:
        batch = queue.get()
        if batch is None:
            break
        else:
            tokenized_batch,orientation_fixed_sequences = batch
            sequences += orientation_fixed_sequences

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
    # sequence_genereator = FILE_SEQUENCE_GENERATOR[file_type]
    # sequence_gen = sequence_genereator(args.sequences, args.max_input_size)
    # sequences = [i for j in sequence_gen for i in j]


    results = post_process_results(args,predictions, args.chain_type, config, sequences)

    # Save results
    save_results(results, args.save_path, file_name, sequences,args.chain_type)
    logging.info(f"Processed Results Saved Successfully at {args.save_path + file_name + '_alignairr_results.csv'}")


if __name__ == '__main__':
    main()
