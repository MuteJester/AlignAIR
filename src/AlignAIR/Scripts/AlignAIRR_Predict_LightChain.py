import sys

# Let's say your module is in '/path/to/your/module'
module_dir = '/home/bcrlab/thomas/AlignAIRR/'

# Append this directory to sys.path
if module_dir not in sys.path:
    sys.path.append(module_dir)

import argparse
import pickle
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from Models.HeavyChain import HeavyChainAlignAIRR
from Models.LightChain import LightChainAlignAIRR
from Trainers import Trainer
from Data import HeavyChainDataset,LightChainDataset

with open('./SequenceSimulation/data/LightChain_KAPPA_DataConfig.pkl','rb') as h:
    lightchain_kappa_config = pickle.load(h)
    

with open('./SequenceSimulation/data/LightChain_LAMBDA_DataConfig.pkl','rb') as h:
    lightchain_lambda_config = pickle.load(h)
    
tokenizer_dictionary = {
    "A": 1,
    "T": 2,
    "G": 3,
    "C": 4,
    "N": 5,
    "P": 0,  # pad token
}



class DynamicConfidenceThreshold:
    def __init__(self, heavy_dataconfig=None, kappa_dataconfig=None, lambda_dataconfig=None):

        self.heavy_dataconfig = heavy_dataconfig
        self.kappa_dataconfig = kappa_dataconfig
        self.lambda_dataconfig = lambda_dataconfig

        self.chain = self.determine_chain()
        self.derive_allele_dictionaries()
        self.derive_call_one_hot_representation()

    def determine_chain(self):
        if self.heavy_dataconfig is not None:
            return 'heavy'
        elif self.kappa_dataconfig is not None and self.lambda_dataconfig is not None:
            return 'light'
        else:
            raise ValueError("Invalid chain configuration")

    def derive_allele_dictionaries(self):

        if self.chain == 'light':
            self.v_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.kappa_dataconfig.v_alleles for j in
                                 self.kappa_dataconfig.v_alleles[i]}
            self.j_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.kappa_dataconfig.j_alleles for j in
                                 self.kappa_dataconfig.j_alleles[i]}

            self.v_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.lambda_dataconfig.v_alleles for j in
                                  self.lambda_dataconfig.v_alleles[i]}
            self.j_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.lambda_dataconfig.j_alleles for j in
                                  self.lambda_dataconfig.j_alleles[i]}
        else:
            self.v_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.v_alleles for j in
                                 self.heavy_dataconfig.v_alleles[i]}
            self.d_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.d_alleles for j in
                                 self.heavy_dataconfig.d_alleles[i]}
            self.j_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.j_alleles for j in
                                 self.heavy_dataconfig.j_alleles[i]}

    def derive_call_one_hot_representation(self):
        if self.chain == 'light':

            v_alleles = sorted(list(self.v_kappa_dict)) + sorted(list(self.v_lambda_dict))
            j_alleles = sorted(list(self.j_kappa_dict)) + sorted(list(self.j_lambda_dict))

            v_allele_count = len(v_alleles)
            j_allele_count = len(j_alleles)

            v_allele_call_ohe = {i: f for i, f in enumerate(v_alleles)}
            j_allele_call_ohe = {i: f for i, f in enumerate(j_alleles)}

            self.properties_map = {
                "V": {"allele_count": v_allele_count, "allele_call_ohe": v_allele_call_ohe},
                "J": {"allel    e_count": j_allele_count, "allele_call_ohe": j_allele_call_ohe},
            }
        else:
            v_alleles = sorted(list(self.v_heavy_dict))
            d_alleles = sorted(list(self.d_heavy_dict))
            d_alleles = d_alleles + ['Short-D']
            j_alleles = sorted(list(self.j_heavy_dict))

            v_allele_count = len(v_alleles)
            d_allele_count = len(d_alleles)
            j_allele_count = len(j_alleles)

            v_allele_call_ohe = {i: f for i, f in enumerate(v_alleles)}
            d_allele_call_ohe = {i: f for i, f in enumerate(d_alleles)}
            j_allele_call_ohe = {i: f for i, f in enumerate(j_alleles)}

            self.properties_map = {
                "V": {"allele_count": v_allele_count, "allele_call_ohe": v_allele_call_ohe},
                "D": {"allele_count": d_allele_count, "allele_call_ohe": d_allele_call_ohe},
                "J": {"allele_count": j_allele_count, "allele_call_ohe": j_allele_call_ohe},
            }

    def __getitem__(self, gene):
        return self.properties_map[gene.upper()]['allele_call_ohe']

    def dynamic_cumulative_confidence_threshold(self, prediction, percentage=0.9):
        sorted_indices = np.argsort(prediction)[::-1]
        selected_labels = []
        cumulative_confidence = 0.0

        total_confidence = sum(prediction)
        threshold = percentage * total_confidence

        for idx in sorted_indices:
            cumulative_confidence += prediction[idx]
            selected_labels.append(idx)

            if cumulative_confidence >= threshold:
                break

        return selected_labels

    def get_alleles(self, likelihood_vectors, confidence=0.9, allele='v', n_process=1):
        def process_vector(vec):
            selected_alleles_index = self.dynamic_cumulative_confidence_threshold(vec, percentage=confidence)
            return [self[allele][i] for i in selected_alleles_index]

        results = Parallel(n_jobs=n_process)(delayed(process_vector)(vec) for vec in tqdm(likelihood_vectors))
        return results

    def agreement_score(self, allele_true, allele_pred):
        """
        Calculate the agreement score between true and predicted alleles.
        """
        set1 = set(allele_true)
        set2 = set(allele_pred)
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        if not union:
            return 0.0  # Avoid division by zero if both lists are empty

        jaccard_index = len(intersection) / len(union)
        return jaccard_index

    def agreement(self, allele_true, allele_pred):
        return len(set(allele_true) & set(allele_pred)) > 0

    def optimize_confidence(self, likelihood_vectors, ground_truth, allele='v', n_process=1, steps=10):
        """
        Find the optimal confidence value.
        """
        best_confidence = 0
        best_score = 0
        best_allele_count = float('inf')

        for confidence in np.linspace(0, 1, steps):
            predicted_alleles = self.get_alleles(likelihood_vectors, confidence, allele, n_process)
            agreement_scores = []

            for pred, true in zip(predicted_alleles, ground_truth):
                score = self.agreement_score(true, pred)
                agreement_scores.append(score)

            average_score = np.mean(agreement_scores)
            average_allele_count = np.mean([len(a) for a in predicted_alleles])

            if average_score > best_score or (average_score == best_score and average_allele_count < best_allele_count):
                best_score = average_score
                best_confidence = confidence
                best_allele_count = average_allele_count

        return best_confidence, best_score, best_allele_count

    def get_confidence_range(self, likelihood_vectors, ground_truth, allele='v', n_process=1, steps=10):
        agg_results = dict()
        for confidence in np.linspace(0, 1, steps):
            predicted_alleles = self.get_alleles(likelihood_vectors, confidence, allele, n_process)
            agreement_scores = []
            calls = []
            agreements = []

            for pred, true in zip(predicted_alleles, ground_truth):
                score = self.agreement_score(true, pred)

                agreements.append(self.agreement(true, pred))
                agreement_scores.append(score)
                calls.append(len(pred))
            agg_results[confidence] = {'agreement_scores': agreement_scores,
                                       'agreements': agreements,
                                       'calls': calls}

        return agg_results


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




def main():
    parser = argparse.ArgumentParser(description='Model Prediction Script')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file for prediction')
    parser.add_argument('--save_path', type=str, required=True, help='Where to save the predictions')
    parser.add_argument('--saved_segment', type=bool, default=False, help='Set to True to save each segment separately')

    args = parser.parse_args()

    lcd = LightChainDataset(data_path=args.csv_file,
                        lambda_dataconfig=lightchain_lambda_config,kappa_dataconfig=lightchain_kappa_config,batch_read_file=True)
    
    print('Init Model..','> ',args.model_checkpoint)
    
    trainer = Trainer(
        model= LightChainAlignAIRR,
        dataset=lcd,
        epochs=1,
        steps_per_epoch=1,
        verbose=1,
    )

    
    trainer.model.build({'tokenized_sequence': (512, 1)})
    
    MODEL_CHECKPOINT = args.model_checkpoint
    print('Loading: ',MODEL_CHECKPOINT.split('/')[-1])
    trainer.model.load_weights(
                MODEL_CHECKPOINT)
    model = trainer.model
    print('Model Loaded!')

    sep = ',' if '.csv' in args.csv_file else '\t'

    try:
        data = pd.read_csv(args.csv_file, usecols=['sequence','v_call','j_call'], sep=sep)
    except ValueError:
        data = pd.read_csv(args.csv_file, usecols=['sequence','v_allele','j_allele'], sep=sep)
        data = data.rename(columns={'v_allele':'v_call','j_allele':'j_call'})

    file_name = args.csv_file.split('/')[-1].split('.')[0]

    if args.save_path == 'None':
        args.save_path = args.model_checkpoint.split('saved_models')[0]

    eval_dataset_ = process_csv_and_tokenize(data['sequence'].to_list(), 512, tokenizer_dictionary)
    print('Train Dataset Encoded!', eval_dataset_.shape)

    padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.uint8)
    dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
        'tokenized_sequence': padded_seqs_tensor})
    dataset = (
        dataset_from_tensors
        .batch(512 * 20)
        .prefetch(tf.data.AUTOTUNE)
    )

    raw_predictions = []

    for i in tqdm(dataset):
        pred = trainer.model.predict(i, verbose=False, batch_size=256)
        raw_predictions.append(pred)

    mutation_rate, v_allele, j_allele = [], [], []
    v_segment, j_segment = [],[]
    type_ = []
    for i in raw_predictions:
        mutation_rate.append(i['mutation_rate'])
        v_allele.append(i['v_allele'])
        j_allele.append(i['j_allele'])
        type_.append(i['type'])
        v_segment.append(i['v_segment'])
        j_segment.append(i['j_segment'])

    mutation_rate = np.vstack(mutation_rate)
    v_allele = np.vstack(v_allele)
    j_allele = np.vstack(j_allele)
    type_ = np.vstack(type_)
    v_segment = np.vstack(v_segment).astype(np.float16)
    j_segment = np.vstack(j_segment).astype(np.float16)


    with open(args.save_path + file_name + '_alignairr_prediction.pkl', 'wb') as h:
                    pickle.dump({
                        'v_proba':v_allele,
                        'j_proba':j_allele,
                        'mutation_rate': mutation_rate,
                        'type':type_,
                    }, h)
                    
    if args.saved_segment:
        with open(args.save_path + file_name + '_v_segment.pkl', 'wb') as h:
            pickle.dump(v_segment, h)
        with open(args.save_path + file_name + '_j_segment.pkl', 'wb') as h:
            pickle.dump(j_segment, h)
                    
if __name__ == "__main__":
    main()
