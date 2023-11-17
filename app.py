import pandas as pd
from flask import Flask,render_template,request
from airrship.create_repertoire import generate_sequence_spesific
import numpy as np
import tensorflow as tf
from UnboundedTrainer import SingleBeamUnboundedTrainer
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeam2,VDeepJAllignExperimentalSingleBeamRG
from VDeepJUnbondedDataset import global_genotype,load_data
import random
from flask_ngrok import run_with_ngrok

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
from VDeepJUnbondedDataset import VDeepJUnbondedDataset
model = VDeepJAllignExperimentalSingleBeamRG
trainer = SingleBeamUnboundedTrainer(model,epochs=10,steps_per_epoch=100,batch_size=32,verbose=True,batch_file_reader=True)
trainer.model.build({'tokenized_sequence':(512,1),'tokenized_sequence_for_masking':(512,1)})
#trainer.model.load_weights('E:\\Immunobiology\\AlignAIRR\\\\sf5_unbounded_experimentalv4_model')
trainer.model.load_weights('E:/Immunobiology/AlignAIRR/sf5_unbounded_experimental_mh_single_beam_RG_end_corrected_model')


def log_threshold(prediction, th=0.15):
    ast = np.argsort(prediction)[::-1]
    R = [ast[0]]
    for ip in range(1, len(ast)):
        DIFF = (prediction[ast[ip - 1]] / prediction[ast[ip]])
        if DIFF < th:
            R.append(ast[ip])
        else:
            break
    return R


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



def extract_prediction_alleles(probabilites, th=0.4):
    V_ratio = []
    for v_all in (probabilites):
        v_alleles = log_threshold(v_all, th=th)
        V_ratio.append([v_allele_call_rev_ohe[i] for i in v_alleles])
    return V_ratio


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


class HTMLSequenceGate:
    def __init__(self, airship_sequence):
        self.mutated_seq = airship_sequence.mutated_seq
        self.html_seq = []
        self.v_seq_start = airship_sequence.v_seq_start
        self.v_seq_end = airship_sequence.v_seq_end
        self.d_seq_start = airship_sequence.d_seq_start
        self.d_seq_end = airship_sequence.d_seq_end
        self.j_seq_start = airship_sequence.j_seq_start
        self.j_seq_end = airship_sequence.j_seq_end
        self.mutations = airship_sequence.mutations

        self.v_color = 'green'
        self.d_color = 'fuchsia'
        self.j_color = 'blue'
        self.mutation_color = 'red'
        self.junction_color = 'black'

    def get_colored_v(self):
        for i in self.mutated_seq[self.v_seq_start:self.v_seq_end]:
            self.html_seq.append(f'<span style="color: {self.v_color};">{i}</span>')

    def get_colored_d(self):
        for i in self.mutated_seq[self.d_seq_start:self.d_seq_end]:
            self.html_seq.append(f'<span style="color: {self.d_color};">{i}</span>')

    def get_colored_j(self):
        for i in self.mutated_seq[self.j_seq_start:self.j_seq_end]:
            self.html_seq.append(f'<span style="color: {self.j_color};">{i}</span>')

    def get_colored_mutation(self):
        for pos in self.mutations:
            self.html_seq[pos-1] = f'<span style="color: {self.mutation_color};">{self.mutated_seq[pos-1]}</span>'

    def get_colored_sequence(self):
        self.get_colored_v()
        for i in self.mutated_seq[self.v_seq_end:self.d_seq_start]:
            self.html_seq.append(f'<span style="font-weight: bold;color: {self.junction_color};">{i}</span>')
        self.get_colored_d()
        for i in self.mutated_seq[self.d_seq_end:self.j_seq_start]:
            self.html_seq.append(f'<span style="font-weight: bold;color: {self.junction_color};">{i}</span>')
        self.get_colored_j()
        self.get_colored_mutation()

        return ''.join(self.html_seq)





app = Flask(__name__)
run_with_ngrok(app)

@app.route('/', methods=['POST','GET'])
def handle_generate_button_click():
    v = request.form.get('VAlleleSelect')
    d = request.form.get('DAlleleSelect')
    j = request.form.get('JAlleleSelect')
    if v is None:
        v = random.choice(locus[0]['V']).ungapped_seq.upper()
    if d is None:
        d = random.choice(locus[0]['D']).ungapped_seq.upper()
    if j is None:
        j = random.choice(locus[0]['J']).ungapped_seq.upper()

    # Process the selected alleles as needed

    # Assuming you have a list of letters and their corresponding colors
    generated_sequence = generate_sequence_spesific(rv_dict[v], rd_dict[d],
                                                         rj_dict[j], data_dict, mutate=True,
                                                         mutation_rate=float(
                                                             0.5))

    predicted = predict_sample(generated_sequence.mutated_seq)
    start, end = getting_padding_size(generated_sequence.mutated_seq)
    pad = start
    pred = [int(np.round(predicted[i].flatten()[0], 0)) - pad for i in
            ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']]

    V = extract_prediction_alleles(predicted['v_allele'])
    D = d_allele_call_rev_ohe[np.argmax(predicted['d_allele'].flatten())]
    J = j_allele_call_rev_ohe[np.argmax(predicted['j_allele'].flatten())]

    generated_sequence.v_seq_start -= 1
    generated_sequence.v_seq_end -= 1
    generated_sequence.d_seq_start -= 1
    generated_sequence.d_seq_end -= 1
    generated_sequence.j_seq_start -= 1
    generated_sequence.j_seq_end -= 1

    generated_sequence_html = HTMLSequenceGate(generated_sequence).get_colored_sequence()
    prediction_table = {
        'V Start':{'GT':generated_sequence.v_seq_start,'AR':pred[0]},
        'V End': {'GT': generated_sequence.v_seq_end, 'AR': pred[1]},
        'D Start': {'GT': generated_sequence.d_seq_start, 'AR': pred[2]},
        'D End': {'GT': generated_sequence.d_seq_end, 'AR': pred[3]},
        'J Start': {'GT': generated_sequence.j_seq_start, 'AR': pred[4]},
        'J End': {'GT': generated_sequence.j_seq_end, 'AR': pred[5]},
        'V Allele': {'GT': generated_sequence.v_allele.name, 'AR': V},
        'D Allele': {'GT': generated_sequence.d_allele.name, 'AR': D},
        'J Allele': {'GT': generated_sequence.j_allele.name, 'AR': J}
        }

    return render_template('MainPage.html',v_allele_list = v_dict,d_allele_list = d_dict,j_allele_list = j_dict,
                           generated_sequence = generated_sequence_html,prediction_table=prediction_table)

@app.route('/predict', methods=['POST'])
def handle_predict_button_click():
    # Your prediction logic here
    # For example:
    sequence = request.form.get('sequence')  # Assuming you send the sequence as part of the form
    print(sequence)
    #prediction = predict_sequence(sequence)  # Assuming you have a function to predict the sequence

    # Return the prediction or render a template with the prediction
    #return render_template('PredictionPage.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host='147.235.219.65')
