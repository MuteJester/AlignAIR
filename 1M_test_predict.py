import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,7) 
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import requests
import json
import numpy as np
from airrship.create_repertoire import create_allele_dict
import importlib
import os
import matplotlib as mpl
from collections import defaultdict

import numpy as np

mpl.rcParams['figure.figsize'] = (20,11)
sns.set_context('poster')
tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
max_length=512

import pickle

def _process_and_dpad(sequence, train=True):
    start, end = None, None
    trans_seq = [tokenizer_dictionary[i] for i in sequence]

    gap = max_length - len(trans_seq)
    iseven = gap % 2 == 0
    whole_half_gap = gap // 2

    if iseven:
        trans_seq = [0] * whole_half_gap + trans_seq + ([0] * whole_half_gap)
        if train:
            start, end = whole_half_gap, max_length - whole_half_gap - 1

    else:
        trans_seq = [0] * (whole_half_gap + 1) + trans_seq + ([0] * whole_half_gap)
        if train:
            start, end = (whole_half_gap + 1, max_length - whole_half_gap - 1)

    return trans_seq, start, end if iseven else (end + 1)

def process_sequences(self, data: pd.DataFrame, corrupt_beginning=False, verbose=False):
    padded_sequences = []
    v_start, v_end, d_start, d_end, j_start, j_end = [], [], [], [], [], []
    iterator = tqdm(data.itertuples(), total=len(data)) if verbose else data.itertuples()
    for row in iterator:
        seq = row.sequence
        padded_array, start, end = _process_and_dpad(seq, self.max_length)
        padded_sequences.append(padded_array)
        _adjust = start


        v_start.append(start)
        j_end.append(end)
        v_end.append(row.v_sequence_end + _adjust)
        d_start.append(row.d_sequence_start + _adjust)
        d_end.append(row.d_sequence_end + _adjust)
        j_start.append(row.j_sequence_start + _adjust)

    v_start = np.array(v_start)
    v_end = np.array(v_end)
    d_start = np.array(d_start)
    d_end = np.array(d_end)
    j_start = np.array(j_start)
    j_end = np.array(j_end)

    padded_sequences = np.vstack(padded_sequences)

    return v_start, v_end, d_start, d_end, j_start, j_end, padded_sequences
def global_genotype():
    try:
        path_to_data = importlib.resources.files(
            'airrship').joinpath("data")
    except AttributeError:
        with importlib.resources.path('airrship', 'data') as p:
            path_to_data = p
    v_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHV.fasta")
    d_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHD.fasta")
    j_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHJ.fasta")

    vdj_allele_dicts = {"V": v_alleles,
                        "D": d_alleles,
                        "J": j_alleles}

    chromosome1, chromosome2 = defaultdict(list), defaultdict(list)
    for segment in ["V", "D", "J"]:
        allele_dict = vdj_allele_dicts[segment]
        for gene in allele_dict.values():
            for allele in gene:
                chromosome1[segment].append(allele)
                chromosome2[segment].append(allele)

    locus = [chromosome1, chromosome2]
    return locus


def decompose_call(call):
    family, G = call.split("-", 1)
    gene, allele = G.split("*")
    return family, gene, allele
locus = global_genotype()

v_dict = dict()
for call in ["V"]:
    for idx in range(2):
        for N in locus[idx][call]:
            if call == "V":
                family, G = N.name.split("-", 1)
                gene, allele = G.split("*")
                v_dict[N.name] = {
                    "family": family,
                    "gene": gene,
                    "allele": allele,
                }

v_families = sorted(set([v_dict[i]["family"] for i in v_dict]))
v_genes = sorted(set([v_dict[i]["gene"] for i in v_dict]))
v_alleles = sorted(set([v_dict[i]["allele"] for i in v_dict]))
v_family_call_ohe = {f: i for i, f in enumerate(v_families)}
v_gene_call_ohe = {f: i for i, f in enumerate(v_genes)}
v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}

from VDeepJUnbondedDataset import global_genotype

locus = global_genotype()
v_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['V']}
d_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['D']}
j_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['J']}
        
v_alleles = sorted(list(v_dict))
d_alleles = sorted(list(d_dict))
d_alleles = d_alleles + ['Short-D']
j_alleles = sorted(list(j_dict))

v_allele_count = len(v_alleles)
d_allele_count = len(d_alleles)
j_allele_count = len(j_alleles)


v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

v_allele_call_rev_ohe = {i: f for i, f in enumerate(v_alleles)}
d_allele_call_rev_ohe = {i: f for i, f in enumerate(d_alleles)}
j_allele_call_rev_ohe = {i: f for i, f in enumerate(j_alleles)}

def encode_igb_v_call(v_call):
    v = np.zeros(len(v_allele_call_rev_ohe))
    for i in v_call.split(','):
        v[v_allele_call_ohe[i]] = 1
    return v
label_num_sub_classes_dict = {
    "V": {
        "family": v_family_call_ohe,
        "gene": v_gene_call_ohe,
        "allele": v_allele_call_ohe}
}

import tensorflow as tf
from multiprocessing import Pool, cpu_count

tokenizer_dictionary = {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }

max_seq_length = 512
def _process_and_dpad(sequence, train=True):
    """
    Private method, converts sequences into 4 one hot vectors and paddas them from both sides with zeros
    equal to the diffrenece between the max length and the sequence length
    :param nuc:
    :param self.max_seq_length:
    :return:
    """

    start, end = None, None
    trans_seq = [tokenizer_dictionary[i] for i in sequence]

    gap = max_seq_length - len(trans_seq)
    iseven = gap % 2 == 0
    whole_half_gap = gap // 2

    if iseven:
        trans_seq = [0] * whole_half_gap + trans_seq + ([0] * whole_half_gap)
        if train:
            start, end = whole_half_gap, max_seq_length - whole_half_gap - 1

    else:
        trans_seq = [0] * (whole_half_gap + 1) + trans_seq + ([0] * whole_half_gap)
        if train:
            start, end = (
                whole_half_gap + 1,
                max_seq_length - whole_half_gap - 1,
            )

    return trans_seq, start, end if iseven else (end + 1)

def preprocess_data(data):
    # Implement your data preprocessing here
    # This function should take an individual data sample or a batch of data
    # and return the preprocessed data
    return np.vstack([_process_and_dpad(i) for i in data])
    
def create_dataset(data, batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
def predict_in_batches(model, dataset):
    raw_predictions = []
    for batch_data in dataset:
        batch_preds = model.predict(batch_data, verbose=True)
        raw_predictions.extend(batch_preds)
    return raw_predictions
def log_threshold(prediction,th=0.4):
    ast = np.argsort(prediction)[::-1]
    R = [ast[0]]
    for ip in range(1,len(ast)):
        DIFF = np.log(prediction[ast[ip-1]]/prediction[ast[ip]])
        if DIFF<th:
            R.append(ast[ip])
        else:
            break
    return R
def log_threshold_r21(prediction,th=0.4):
    ast = np.argsort(prediction)[::-1]
    R = [ast[0]]
    for ip in range(1,len(ast)):
        DIFF = np.log(prediction[ast[0]]/prediction[ast[ip]])
        if DIFF<th:
            R.append(ast[ip])
        else:
            break
    return R
def extract_prediction_alleles(probabilites,th=0.4):
    V_ratio = []
    for v_all in tqdm(probabilites):
        v_alleles  = log_threshold(v_all,th=th)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio
def extract_prediction_alleles_norm(probabilites,th=0.4):
    V_ratio = []
    for v_all in tqdm(probabilites):
        v_alleles  = log_threshold(v_all/v_all.max(),th=th)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio
def extract_prediction_alleles_r21(probabilites,th=0.4):
    V_ratio = []
    for v_all in tqdm(probabilites):
        v_alleles  = log_threshold_r21(v_all/v_all.max(),th=th)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio
def dynamic_cumulative_confidence_threshold(prediction, percentage=0.9):
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
def extract_prediction_alleles_dynamic_sum(probabilites,percentage=0.9,type='v'):
    V_ratio = []
    for v_all in tqdm(probabilites):
        v_alleles  = dynamic_cumulative_confidence_threshold(v_all,percentage=percentage)
        if type == 'v':
            V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
        elif type =='d':
            V_ratio.append([d_allele_call_rev_ohe[i] for i in v_alleles])
        elif type == 'j':
            V_ratio.append([j_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio

def binary_search_entropy_threshold(prediction, min_entropy=0.2):
    sorted_indices = np.argsort(prediction)[::-1]
    
    def calculate_entropy(chunk):
        probs = prediction[chunk]
        entropy = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)
        return np.sum(entropy)
    
    def search_threshold(left, right):
        mid = (left + right) // 2
        left_chunk = sorted_indices[:mid]
        right_chunk = sorted_indices[mid:]
        
        left_entropy = calculate_entropy(left_chunk)
        right_entropy = calculate_entropy(right_chunk)
        
        if left_entropy <= right_entropy:
            return left_chunk, left_entropy
        else:
            return right_chunk, right_entropy
    
    selected_labels = []
    left, right = 0, len(sorted_indices)
    
    while left < right:
        chunk, entropy = search_threshold(left, right)
        selected_labels.extend(chunk)
        
        if entropy <= min_entropy:
            break
        
        if len(chunk) == right:
            break  # Avoid an infinite loop in the unlikely event of no entropy improvement
        
        if len(chunk) == left:
            left += 1  # Avoid an infinite loop in the unlikely event of no entropy improvement
        else:
            left, right = len(selected_labels), len(sorted_indices)
    
    return selected_labels
def extract_prediction_alleles_entropy_sum(probabilites,min_entropy=0.2):
    V_ratio = []
    for v_all in tqdm(probabilites):
        v_alleles  = binary_search_entropy_threshold(v_all,min_entropy=min_entropy)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio
def gini_impurity_threshold(prediction, min_impurity=0.2):
    # Sort the probabilities in descending order and get the sorted indices
    sorted_indices = np.argsort(prediction)[::-1]
    selected_labels = []
    
    # Initialize cumulative Gini impurity and threshold
    cumulative_impurity = 0.0
    threshold = 0.0  # Initialize threshold to 0.0
    
    for idx in sorted_indices:
        label_prob = prediction[idx]
        
        # Calculate the Gini impurity of the current label
        label_impurity = 2 * label_prob * (1 - label_prob)
        
        # Add the label's impurity to the cumulative impurity
        cumulative_impurity += label_impurity
        
        # Calculate the threshold as a function of cumulative impurity
        threshold = cumulative_impurity / len(sorted_indices)
        
        selected_labels.append(idx)
        
        # Stop when the threshold reaches the minimum impurity
        if threshold >= min_impurity:
            break
    
    return selected_labels
def extract_prediction_alleles_gini(probabilites,min_impurity=0.2):
    
    V_ratio = []
    for v_all in tqdm(probabilites):
        v_alleles  = gini_impurity_threshold(v_all,min_impurity=min_impurity)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio
def _process_and_dpad(sequence, max_seq_length, tokenizer_dictionary):
    trans_seq = [tokenizer_dictionary.get(i, 0) for i in sequence]  # Use .get() to handle unknown characters
    gap = max_seq_length - len(trans_seq)
    iseven = gap % 2 == 0
    whole_half_gap = gap // 2

    if iseven:
        trans_seq = [0] * whole_half_gap + trans_seq + ([0] * whole_half_gap)
    else:
        trans_seq = [0] * (whole_half_gap + 1) + trans_seq + ([0] * whole_half_gap)

    return trans_seq


def tokenize_chunk(chunk, max_seq_length, tokenizer_dictionary):
    return [(index, _process_and_dpad(sequence, max_seq_length, tokenizer_dictionary)) for index, sequence in chunk]


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



import tensorflow as tf
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRF,VDeepJAllignExperimentalSingleBeamConvSegmentationResidual_DC_MR_ONLY_ALLELES
from Trainer import SingleBeamSegmentationTrainerV1__5
trainer = SingleBeamSegmentationTrainerV1__5(
    model=VDeepJAllignExperimentalSingleBeamConvSegmentationResidual_DC_MR_ONLY_ALLELES,
    data_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/AlignAIRR_Large_Train_Dataset.csv",
    batch_read_file=True,
    epochs=1,
    batch_size=1,
    steps_per_epoch=150_000,
    verbose=1,
    corrupt_beginning=True,
    classification_head_metric=[tf.keras.metrics.AUC(),tf.keras.metrics.AUC(),tf.keras.metrics.AUC()],
    interval_head_metric=tf.keras.losses.mae,
    corrupt_proba=0.7,
    airrship_mutation_rate=0.25,
    nucleotide_add_coef=210,
    nucleotide_remove_coef=330,
    random_sequence_add_proba=0.45,
    single_base_stream_proba=0.05,
    duplicate_leading_proba=0.25,
    random_allele_proba=0.25,
    num_parallel_calls=32,
)


trainer.model.build({'tokenized_sequence':(512,1)})
#trainer.model.load_weights("/localdata/alignairr_data/sf5_alignairr_segmentation_residual_s_v_d_j_embedding_product/saved_models/sf5_alignairr_segmentation_residual_s_v_d_j_embedding_product_cp1")
trainer.model.load_weights("/localdata/alignairr_data/VDeepJAllignExperimentalSingleBeamConvSegmentationResidual_DC_MR/evaluation_checkpoints/model_acc_0.9946/variables/variables")
print('Model Loaded!')


PATH ='/localdata/alignairr_data/test_data_flat_usage/'
datasets_files = os.listdir(PATH)
datasets_files = list(filter(lambda x: '.tsv' in x , datasets_files))
datasets_files = list(filter(lambda x: 'add_n' in x , datasets_files))
tokenizer_dictionary =  {
            "A": 1,
            "T": 2,
            "G": 3,
            "C": 4,
            "N": 5,
            "P": 0,  # pad token
        }
results = {'v_call':dict(),'d_call':dict(),'j_call':dict()}

for pred_file in tqdm(datasets_files):
    # Read the data from the TSV file
    data = pd.read_table(PATH+pred_file,usecols=['sequence','v_call','d_call','j_call'])

    eval_dataset_ = process_csv_and_tokenize(data['sequence'],512,tokenizer_dictionary)
    
    
    print('Train Dataset Encoded!')
    padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.uint8)
    dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
        'tokenized_sequence': padded_seqs_tensor})
    dataset = (
        dataset_from_tensors
        .batch(512*20)
        .prefetch(tf.data.AUTOTUNE)
    )

    raw_predictions =[]

    for i in tqdm(dataset):
        pred = trainer.model.predict(i, verbose=False,batch_size=512)
        raw_predictions.append(pred)
        
    mutation_rate,v_allele,d_allele,j_allele = [],[],[],[]
    for i in raw_predictions:
        mutation_rate.append(i['mutation_rate'])
        v_allele.append(i['v_allele'])
        d_allele.append(i['d_allele'])
        j_allele.append(i['j_allele'])
    mutation_rate = np.vstack(mutation_rate)
    v_allele = np.vstack(v_allele)
    d_allele = np.vstack(d_allele)
    j_allele = np.vstack(j_allele)
    
    allele_m = {'v_call':v_allele,'d_call':d_allele,'j_call':j_allele}
    
    for call in ['v_call','d_call','j_call']:
        X = extract_prediction_alleles_dynamic_sum(allele_m[call],percentage=0.95,type=call.split('_')[0])
        hits = [len( set(i.split(',')) &set(j)) > 0 for i,j in zip(data[call],X)]
        agg = sum(hits)/len(hits)
        
        above_10 = len(list(filter(lambda x: x>10,list(map(len,X)))))
        above_5= len(list(filter(lambda x: x>10,list(map(len,X)))))
        above_3= len(list(filter(lambda x: x>10,list(map(len,X)))))
        above_2= len(list(filter(lambda x: x>10,list(map(len,X)))))

        results[call][pred_file] = {'agreement':agg,'above_10':above_10,'above_5':above_5,'above_3':above_3,'above_2':above_2,
                                    'mean_mutation_rate':np.mean(mutation_rate),'std_mutation_rate':np.std(mutation_rate)}
    print(results)
    print('===========================================================================')
    print('XVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVXVX')
    print('===========================================================================')

with open('/localdata/alignairr_data/VDeepJAllignExperimentalSingleBeamConvSegmentationResidual_DC_MR/'+'test_data_flat_usage_VDeepJAllignExperimentalSingleBeamConvSegmentationResidual_DC_MR_result.pkl','wb') as h:
    pickle.dump(results,h)