import csv

from flask import redirect, send_from_directory, Response
import numpy as np
import tensorflow as tf
from airrship.create_repertoire import generate_sequence_spesific
from flask import Flask, request, jsonify
from flask import redirect
from flask_cors import CORS
from sklearn.decomposition import PCA,FactorAnalysis,FastICA,TruncatedSVD
from sklearn.manifold import TSNE,MDS
from umap import UMAP
from UnboundedTrainer import SingleBeamUnboundedTrainer
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamRG
from VDeepJUnbondedDataset import global_genotype, load_data
import pandas as pd
import re
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os

locus = global_genotype()
v_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['V']}
d_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['D']}
j_dict = {i.name: i.ungapped_seq.upper() for i in locus[0]['J']}
v_dict ={i:v_dict[i] for i in sorted(list(v_dict),key=lambda x: x.split('-')[-1])}
d_dict ={i:d_dict[i] for i in sorted(list(d_dict),key=lambda x: x.split('-')[-1])}
j_dict ={i:j_dict[i] for i in sorted(list(j_dict),key=lambda x: x.split('-')[-1])}
rv_dict = {i.ungapped_seq.upper():i for i in locus[0]['V']}
rd_dict = {i.ungapped_seq.upper():i for i in locus[0]['D']}
rj_dict = {i.ungapped_seq.upper():i for i in locus[0]['J']}
nv_dict = {i.name: i for i in locus[0]['V']}
nd_dict = {i.name: i for i in locus[0]['D']}
nj_dict = {i.name: i for i in locus[0]['J']}

data_dict = load_data()

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


#load model
model = VDeepJAllignExperimentalSingleBeamRG
trainer = SingleBeamUnboundedTrainer(model,epochs=10,steps_per_epoch=100,batch_size=32,verbose=True,batch_file_reader=True)
trainer.model.build({'tokenized_sequence':(512,1),'tokenized_sequence_for_masking':(512,1)})
trainer.model.load_weights('E:/Immunobiology/AlignAIRR/archive/sf5_alignairr_latest_mh_single_beam_RG_end_corrected_model')


# Get V Latent + Dimension Reducer
import pickle
with open('E:/Immunobiology/AlignAIRR/V_Allele_F_dict.pkl','rb') as h:
    V_Alleles = pickle.load(h)
eval_dataset_ = trainer.train_dataset.tokenize_sequences(list(V_Alleles.values()))

v_mask_input_embedding = trainer.model.concatenated_v_mask_input_embedding(
    eval_dataset_
)
v_feature_map = trainer.model._encode_masked_v_signal(v_mask_input_embedding)

v_allele_latent = trainer.model.v_allele_mid(v_feature_map)

dec = PCA(n_components=2)

dec_df = pd.DataFrame(dec.fit_transform(v_allele_latent.numpy()),columns=['d1','d2'])
dec_df['family']= [re.search(r'F[0-9]+',i).group() for i in list(V_Alleles)]



def predict_sample(sample):
    eval_dataset_ = trainer.train_dataset.tokenize_sequences([sample])
    padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.int32)
    dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
        'tokenized_sequence': padded_seqs_tensor,
        'tokenized_sequence_for_masking': padded_seqs_tensor
    })
    dataset = (
        dataset_from_tensors
        .batch(1)
        .prefetch(tf.data.AUTOTUNE)
    )

    predicted = trainer.model.predict(dataset, verbose=True)
    return predicted
def get_v_latent_projection(sequence):
    eval_dataset_ = trainer.train_dataset.tokenize_sequences([sequence])
    v_mask_input_embedding = trainer.model.concatenated_v_mask_input_embedding(
        eval_dataset_
    )
    v_feature_map = trainer.model._encode_masked_v_signal(v_mask_input_embedding)

    v_allele_latent = trainer.model.v_allele_mid(v_feature_map)
    v_allele_latent = dec.transform(v_allele_latent.numpy())
    return v_allele_latent[0]

def getting_padding_size(seq, max_length=512):
    start, end = None, None
    gap = max_length - len(seq)
    iseven = gap % 2 == 0
    whole_half_gap = gap // 2

    if iseven:
        start, end = whole_half_gap, whole_half_gap

    else:
        start, end = whole_half_gap + 1, whole_half_gap
    return start, end
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
    for v_all in (probabilites):
        v_alleles  = log_threshold(v_all,th=th)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio
def allign_sequences(seqs):
    sequences = [
    SeqRecord(Seq(seq), id=f"seq{en}")
    for en,seq in enumerate(seqs)
    ]
    SeqIO.write(sequences, r"C:\Users\Tomas\Downloads\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\sequences.fasta", "fasta")

    # Define Clustal Omega command
    clustalomega_cline = ClustalOmegaCommandline(infile=r"C:\Users\Tomas\Downloads\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\sequences.fasta",
                                                 outfile=r"C:\Users\Tomas\Downloads\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\aligned.fasta", verbose=True, auto=True,force=True)
    clustalomega_cline.program_name = r"C:\Users\Tomas\Downloads\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\clustalo.exe"
    stdout, stderr = clustalomega_cline()

    alignment = AlignIO.read(r"C:\Users\Tomas\Downloads\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\aligned.fasta", "fasta")
    return alignment

app = Flask(__name__, static_url_path='/static')
CORS(app)
#run_with_ngrok(app)

@app.route('/')
def hello():
    return redirect('/static/index.html')

@app.route('/get_alleles', methods=['GET'])
def get_alleles():
    type = request.args.get('type')
    if type== 'V':
        return jsonify({"V": v_alleles})
    elif type=='D':
        return jsonify({"D": d_alleles})
    elif type =='J':
        return jsonify({'J': j_alleles})

@app.route('/get_v_latent', methods=['GET'])
def get_v_latent():
    return jsonify({"x": dec_df['d1'].to_list(),'y':dec_df['d2'].to_list(),'label':dec_df['family'].to_list(),
                    'allele':list(V_Alleles)})

@app.route('/generate_sequence', methods=['POST'])
def generate_sequence():
    request_data = request.get_json()
    V = request_data['V']
    D = request_data['D']
    J = request_data['J']
    generated_sequence = generate_sequence_spesific(nv_dict[V], nd_dict[D],
                                                    nj_dict[J], data_dict, mutate=True,
                                                    mutation_rate=np.random.uniform(0,0.15,1).item())

    response = {
        'sequence':generated_sequence.mutated_seq,
        'v_start':generated_sequence.v_seq_start,
        'v_end': generated_sequence.v_seq_end,
        'd_start': generated_sequence.d_seq_start,
        'd_end': generated_sequence.d_seq_end,
        'j_start': generated_sequence.j_seq_start,
        'j_end': generated_sequence.j_seq_end,
        'mutation_positions':list(generated_sequence.mutations.keys())

    }
    return jsonify(response)

@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    request_data = request.get_json()
    input_seq = request_data['sequence']
    gt_vstart = int(request_data['v_start'])
    gt_vend = int(request_data['v_end'])

    predicted = predict_sample(input_seq)
    start, end = getting_padding_size(input_seq)
    pad = start
    pred = [int(np.round(predicted[i].flatten()[0], 0)) - pad for i in
            ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']]

    V = extract_prediction_alleles(predicted['v_allele'],th=1)
    D = d_allele_call_rev_ohe[np.argmax(predicted['d_allele'].flatten())]
    J = j_allele_call_rev_ohe[np.argmax(predicted['j_allele'].flatten())]

    top_5 = np.argsort(predicted['v_allele'][0])[::-1][:5]
    top_5 = [{'label':v_allele_call_rev_ohe[i],'value':float(np.round(predicted['v_allele'][0][i],5))} for i in top_5]

    latent_v_proj = get_v_latent_projection(input_seq)

    # Top 5 alignment
    #seqs = [{'id':'Your Sequence','data':input_seq}]+[{'id':i['label'],'data':V_Alleles[i['label']]} for i in top_5]
    seqs = [input_seq[gt_vstart:gt_vend]]+[V_Alleles[i['label']] for i in top_5]
    allinged = allign_sequences(seqs)
    allinged = [str(i.seq) for i in allinged]
    seqs = [{'id': 'Your Sequence', 'data': allinged[0]}] + [{'id': i['label'], 'data': allinged[en+1]} for en,i in
                                                           enumerate(top_5)]

    response = {
        'V':V,
        'D':D,
        'J':J,
        'v_start':pred[0],
        'v_end': pred[1],
        'd_start': pred[2],
        'd_end': pred[3],
        'j_start': pred[4],
        'j_end': pred[5],
        'top_5_probas':top_5,
        'v_latent_x':float(latent_v_proj[0]),
        'v_latent_y':float(latent_v_proj[1]),
        'allgined_seqs':seqs,

    }




    return jsonify(response)

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    # Check if the request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read the uploaded file content
    sequences = file.read().decode('utf-8').splitlines()


    # Check if the uploaded data is a list of sequences
    # You can add more checks based on your requirements
    if not sequences:
        return jsonify({'error': 'Invalid file content'}), 400

    eval_dataset_ = trainer.train_dataset.tokenize_sequences(sequences)
    padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.int32)
    dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
        'tokenized_sequence': padded_seqs_tensor,
        'tokenized_sequence_for_masking': padded_seqs_tensor
    })
    dataset = (
        dataset_from_tensors
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )

    predicted = trainer.model.predict(dataset, verbose=True)

    pads = []
    for i in sequences:
        start, end = getting_padding_size(i)
        pads.append(start)
    pads = np.array(pads)

    reg_keys = ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']
    for key in reg_keys:
        predicted[key] = np.round(predicted[key].flatten()-pads,0).astype(int)

    V = extract_prediction_alleles(predicted['v_allele'].squeeze(), th=1)
    D = [d_allele_call_rev_ohe[np.argmax(i)] for i in predicted['d_allele']]
    J = [j_allele_call_rev_ohe[np.argmax(i)] for i in predicted['j_allele']]

    result = pd.DataFrame({
        key:predicted[key] for key in reg_keys
    })

    result['v_call'] = V
    result['d_call'] = D
    result['j_call'] = J
    result['sequence'] = sequences

    # Convert the DataFrame to a CSV string
    csv_data = result.to_csv(index=False)

    # Create a response with the CSV data
    response = Response(csv_data, content_type="text/csv")

    # Set headers to prompt the user to download the file
    response.headers["Content-Disposition"] = "attachment; filename=results.csv"

    return response


if __name__ == '__main__':
    app.run()