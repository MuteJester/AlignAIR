import pandas as pd
import pickle
import numpy as np
import os
from tqdm.auto import tqdm
from VDeepJUnbondedDataset import global_genotype
#from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamRG
#from UnboundedTrainer import SingleBeamUnboundedTrainer
#import tensorflow as tf

locus = global_genotype()
v_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['V']}
d_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['D']}
j_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['J']}
        
v_alleles = sorted(list(v_dict))
d_alleles = sorted(list(d_dict))
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
def extract_prediction_alleles(probabilites,th=0.4):
    V_ratio = []
    for v_all in tqdm(probabilites):
        v_alleles  = log_threshold(v_all,th=th)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio

# igb_predicted = pd.read_table(r"/localdata/alignairr_data/naive_repertoires/naive_sequences_clean.tsv",index_col=0)
# igb_predicted = pd.read_table("/localdata/alignairr_data/1M/sim_data_1M_asc_P05.tsv",usecols=['v_call']) 
predicted_path= '/localdata/alignairr_data/sf5_alignairr_latest_mh_single_beam_RG_end_corrected_pred/'
predicted_files = os.listdir(predicted_path)
predicted_files = [i for i in predicted_files if '.pkl' in i]

results_bank = dict()


for ax,prediction_file in enumerate(predicted_files):
    file = prediction_file.split('__data__')[1].replace('sim_data_1M_asc_','').replace('sim_data_1M_','').replace('_add_n.pkl','')
    validation = prediction_file.split('__data__')[1].replace('.pkl','')
    igb_predicted = pd.read_table('/localdata/alignairr_data/1M_for_test/'+validation+'.tsv',usecols=['v_call']) 

    with open(predicted_path+prediction_file,'rb') as h:
        alignairr_predicted = pickle.load(h) 
    model,rate = file.split('_rate_')
    rate = float(rate[0]+'.'+rate[1:])

    V = extract_prediction_alleles(alignairr_predicted['v_allele'],th=1)

    hits = [len( set(i.split(',')) &set(j)) > 0 for i,j in zip(igb_predicted.v_call,V)]
    v_acc = sum(hits)/len(hits)

    print('Model: ',model)
    print('Rate: ',rate)
    print('V Accuracy: ',v_acc)
    print('File: ',ax,' / ',len(predicted_files))

    if model in results_bank:
        results_bank[model][rate] = v_acc
    else:
        results_bank[model] = {rate:v_acc}
    print('||=||||=||||=||||=||||=||||=||||=||||=||||=||')

with open('/localdata/alignairr_data/sf5_alignairr_latest_mh_single_beam_RG_end_corrected_1M_V_accuracy.pkl','wb') as h:
    pickle.dump(results_bank,h)


model = VDeepJAllignExperimentalSingleBeamRG
trainer = SingleBeamUnboundedTrainer(model,epochs=10,steps_per_epoch=100,batch_size=32,verbose=True,batch_file_reader=True)
trainer.model.build({'tokenized_sequence':(512,1),'tokenized_sequence_for_masking':(512,1)})
#trainer.model.load_weights('E:\\Immunobiology\\AlignAIRR\\\\sf5_unbounded_experimentalv4_model')
trainer.model.load_weights("/localdata/alignairr_data/sf5_unbounded_experimental_model_mh_single_beam_RG_end_corrected/saved_models/sf5_unbounded_experimental_mh_single_beam_RG_end_corrected_model")

print('Model Weights: ')
print(trainer.model.v_start_mid.weights)
print('\n------------------\n')



# predicted_path= '/localdata/alignairr_data/igblast_results/'
# predicted_files = os.listdir(predicted_path)
# predicted_files = [i for i in predicted_files if '.tsv' in i]
# import os
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor

# def read_file(file_name):
#     p = file_name.replace('1M_asc_P05_model_s5f_20', '1M_S5F_20').replace('P05_model_','')
#     igb_predicted = pd.read_table('/localdata/alignairr_data/1M_for_test/' + p, usecols=['v_call']).fillna('X')
#     alignairr_predicted = pd.read_table(predicted_path + file_name, usecols=['v_call']).fillna('X')
#     return igb_predicted, alignairr_predicted

# predicted_path = '/localdata/alignairr_data/igblast_results/'
# predicted_files = [i for i in os.listdir(predicted_path) if '.tsv' in i]

# results_bank = {}

# # Create a ThreadPoolExecutor
# with ThreadPoolExecutor(max_workers=2) as executor:
#     for ax, prediction_file in enumerate(predicted_files):
#         future = executor.submit(read_file, prediction_file)
        
#         try:
#             igb_predicted, alignairr_predicted = future.result()
#         except Exception as exc:
#             print(f"Generated an exception {exc} while reading {prediction_file}")
#             continue

#         hits = [len(set(i.split(',')) & set(j.split(','))) > 0 for i, j in zip(igb_predicted.v_call, alignairr_predicted.v_call)]
#         v_acc = sum(hits) / len(hits)

#         print('V Accuracy: ', v_acc)
#         print('File: ', ax, ' / ', len(predicted_files))

#         results_bank[prediction_file] = v_acc
#         print('||=||||=||||=||||=||||=||||=||||=||||=||||=||')

# with open('/home/bcrlab/thomas/AlignAIRR/IGB_1M_V_accuracy_second_model.pkl','wb') as h:
#     pickle.dump(results_bank,h)





# def predict_sample(sample):
#     eval_dataset_ = trainer.train_dataset.tokenize_sequences([sample])
#     padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.int32)
#     dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
#         'tokenized_sequence': padded_seqs_tensor,
#         'tokenized_sequence_for_masking': padded_seqs_tensor
#     })
#     dataset = (
#         dataset_from_tensors
#         .batch(1)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     predicted = trainer.model.predict(dataset, verbose=True)
#     return predicted



# tdata = pd.read_table("/localdata/alignairr_data/1M_for_test/sim_data_1M_asc_uniform_rate_001.tsv")
# train_sequences = tdata['sequence']
# X_train = vssds.tokenize_sequences(train_sequences)
# padded_seqs_tensor = tf.convert_to_tensor(X_train, dtype=tf.int32)
# dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
#     'tokenized_sequence': padded_seqs_tensor,
#     'tokenized_sequence_for_masking': padded_seqs_tensor
# })
# dataset = (
#     dataset_from_tensors
#     .batch(768)
#     .prefetch(tf.data.AUTOTUNE)
# )

# raw_predictions = trainer.model.predict(dataset, verbose=True)

# #
