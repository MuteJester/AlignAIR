import sys

# Let's say your module is in '/path/to/your/module'
module_dir = '/home/bcrlab/thomas/AlignAIRR/'

# Append this directory to sys.path
if module_dir not in sys.path:
    sys.path.append(module_dir)
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('poster')
mpl.rcParams['figure.figsize']=(20,7)
from tqdm.auto import tqdm
import re
import plotly.express as ex
import importlib
import plotly.io as pio
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
from scipy.stats import entropy
from collections import defaultdict
import tensorflow as tf
import pickle
from Trainers import Trainer
from Models.HeavyChain import HeavyChainAlignAIRR
from SequenceSimulation.utilities.data_config import DataConfig
from SequenceSimulation.sequence import HeavyChainSequence
import random
from Data import HeavyChainDataset
from SequenceSimulation.mutation import Uniform,S5F


with open('./SequenceSimulation/data/HeavyChain_DataConfig_OGRDB.pkl','rb') as h:
    dataconfig = pickle.load(h)
    dc = dataconfig
    
    
v_dict = {i.name: i.ungapped_seq.upper() for i in sorted([i for j in dc.v_alleles for i in dc.v_alleles[j]],
                                key=lambda x: x.name)}
d_dict = {i.name: i.ungapped_seq.upper() for i in sorted([i for j in dc.d_alleles for i in dc.d_alleles[j]],
                                key=lambda x: x.name)}
j_dict = {i.name: i.ungapped_seq.upper() for i in sorted([i for j in dc.j_alleles for i in dc.j_alleles[j]],
                                key=lambda x: x.name)}

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

tokenizer_dictionary = {
    "A": 1,
    "T": 2,
    "G": 3,
    "C": 4,
    "N": 5,
    "P": 0,  # pad token
}




TRAIN_DATASET = "/localdata/alignairr_data/AlignAIRR_Evaluation_Dataset/HeavyChain_OGRDB_AlignAIRR_S5F_5M_No_Corruption.csv"
train_dataset = HeavyChainDataset(data_path=TRAIN_DATASET
                                  ,dataconfig=dataconfig,
                                  batch_size=32,
                                  max_sequence_length=512,
                                  batch_read_file=True)

trainer = Trainer(
    model= HeavyChainAlignAIRR,
    dataset=train_dataset,
    epochs=1,
    steps_per_epoch=1,
    verbose=1
)
trainer.model.build({'tokenized_sequence': (512, 1)})

MODEL_CHECKPOINT = "/localdata/alignairr_data/AlignAIRR_S5F_OGRDB_DConfig/saved_models/AlignAIRR_S5F_OGRDB_DConfig"
print('Loading: ',MODEL_CHECKPOINT.split('/')[-1])
trainer.model.load_weights(
            MODEL_CHECKPOINT)
model = trainer.model
print('Model Loaded!')


# MISC
v_alleles = [i for j in dataconfig.v_alleles for i in dataconfig.v_alleles[j]]
d_alleles = [i for j in dataconfig.d_alleles for i in dataconfig.d_alleles[j]]
j_alleles = [i for j in dataconfig.j_alleles for i in dataconfig.j_alleles[j]]

random_d = random.choice(d_alleles)
random_j = random.choice(j_alleles)


def get_v_latent(seq):
    # STEP 1 : Produce embeddings for the input sequence
    input_seq = model.reshape_and_cast_input(trainer.train_dataset.tokenize_sequences([seq]))
    input_embeddings = model.input_embeddings(input_seq)

    conv_layer_segmentation_d_res = model._forward_pass_segmentation_feature_extraction(input_embeddings)

    # STEP 3 : Flatten The Feature Derived from the 1D conv layers
    concatenated_signals = conv_layer_segmentation_d_res
    concatenated_signals = model.segmentation_feature_flatten(concatenated_signals)
    concatenated_signals = model.initial_feature_map_dropout(concatenated_signals)
    # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
    v_segment, d_segment, j_segment, mutation_rate = model.predict_segments(concatenated_signals)

    reshape_masked_sequence_v = model.v_mask_reshape(v_segment)

    masked_sequence_v = model.v_mask_gate([reshape_masked_sequence_v, input_embeddings])
    # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
    v_feature_map = model._encode_masked_v_signal(masked_sequence_v)

    # ============================ V =============================
    v_allele_middle = model.v_allele_mid(v_feature_map)
    return v_allele_middle.numpy().flatten()
def get_v_latents(seqs):
    # STEP 1 : Produce embeddings for the input sequence
    input_seq = model.reshape_and_cast_input(trainer.train_dataset.tokenize_sequences(seqs))
    input_embeddings = model.input_embeddings(input_seq)

    conv_layer_segmentation_d_res = model._forward_pass_segmentation_feature_extraction(input_embeddings)

    # STEP 3 : Flatten The Feature Derived from the 1D conv layers
    concatenated_signals = conv_layer_segmentation_d_res
    concatenated_signals = model.segmentation_feature_flatten(concatenated_signals)
    concatenated_signals = model.initial_feature_map_dropout(concatenated_signals)
    # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
    v_segment, d_segment, j_segment, mutation_rate = model.predict_segments(concatenated_signals)

    reshape_masked_sequence_v = model.v_mask_reshape(v_segment)

    masked_sequence_v = model.v_mask_gate([reshape_masked_sequence_v, input_embeddings])
    # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
    v_feature_map = model._encode_masked_v_signal(masked_sequence_v)

    # ============================ V =============================
    v_allele_middle = model.v_allele_mid(v_feature_map)
    return v_allele_middle.numpy()

def format_inputs(seq):
    inp = trainer.train_dataset.tokenize_sequences(seq)
    inp = {'tokenized_sequence':inp}
    return inp

def get_v_latents(seqs):
    fseqs = format_inputs(seqs)
    return model.get_v_latent_dimension(fseqs).numpy()

def format_input(seq):
    inp = trainer.train_dataset.tokenize_sequences([seq])
    inp = {'tokenized_sequence':inp}
    return inp


class LatnetAnchor:
    def __init__(self,allele):
        self.sequence = HeavyChainSequence([allele,random_d,random_j],dataconfig)
        self.name = allele.name
        
        inp = trainer.train_dataset.tokenize_sequences([self.sequence.ungapped_seq])
        inp = {'tokenized_sequence':inp}
        self.latent_pos = model.get_v_latent_dimension(inp).numpy().flatten()
        
        
        
print('Starting to Create Latent Anchors')
latnet_anchors =  [LatnetAnchor(i) for i in v_alleles]

print('Starting to Create Random Spots')
random_spots = [format_input(''.join(np.random.choice(['A','T','C','G'],size=np.random.randint(287,302)))) for _ in range(5)]
random_spots = [model.get_v_latent_dimension(i).numpy().flatten() for i in random_spots]





from uuid import uuid4
NREP = 1000
class Agent:
    def __init__(self,lantent_anchor):
        self.sequence = lantent_anchor.sequence
        self.name = lantent_anchor.name
        
    def mutation_field(self,max_mrate,mmodel):
        results = dict()
        _id = str(uuid4())
        for E in range(NREP):
            muts = []
            MMODEL = mmodel(max_mrate,max_mrate)
            self.sequence.mutate(MMODEL)
            sorted_m_index = sorted(self.sequence.mutations)
            seqs = []
            for slc in range(1,len(sorted_m_index)+1):
                c_mseq = list(self.sequence.ungapped_seq)
                for m in sorted_m_index[:slc]:
                    c_mseq[m] = self.sequence.mutations[m][-1]
                seqs.append(''.join(c_mseq))
                muts.append(slc)
            results[E] = {'muts':muts,'latent':get_v_latents(seqs)} 
            
        
        min_mutation_overlap = min([max(results[i]['muts']) for i in results])
        for key in results:
            if max(results[key]['muts']) > min_mutation_overlap:
                diff = results[key]['muts']-min_mutation_overlap
                results[key]['latent'] = results[key]['latent'][:-diff,:]
        
        avg_latent = np.mean([results[i]['latent'] for i in results], axis=0)
        muts = None
        for i in results:
            if len( results[i]['muts']) == min_mutation_overlap:
                muts= results[i]['muts']
        
        return muts,avg_latent,[_id]*len(muts),[self.name]*len(muts)
    
    
print ('Starting Simulation')
agents =  [Agent(i) for i in latnet_anchors]

logs = []
print('Uniform Simulations')
for A in tqdm(agents):
    mutation_counts,latnet,ids,names = A.mutation_field(0.35,Uniform)
    logs.append((mutation_counts,latnet,ids,names))
    
s5f_logs = []
print('S5F Simulations')
for A in tqdm(agents):
    mutation_counts,latnet,ids,names = A.mutation_field(0.4,S5F)
    s5f_logs.append((mutation_counts,latnet,ids,names))
    
    
with open(f'/localdata/alignairr_data/AlignAIRR_S5F_OGRDB_DConfig/{NREP}_s5f_agent_walk_data.pkl','wb') as h:
    pickle.dump(s5f_logs,h)
with open(f'/localdata/alignairr_data/AlignAIRR_S5F_OGRDB_DConfig/{NREP}_s5f_anchor_data.pkl','wb') as h:
    pickle.dump(latnet_anchors,h)

with open(f'/localdata/alignairr_data/AlignAIRR_S5F_OGRDB_DConfig/{NREP}_uniform_agent_walk_data.pkl','wb') as h:
    pickle.dump(logs,h)
with open(f'/localdata/alignairr_data/AlignAIRR_S5F_OGRDB_DConfig/{NREP}_uniform_anchor_data.pkl','wb') as h:
    pickle.dump(latnet_anchors,h)
    
    
    
print('Saved and Done!')