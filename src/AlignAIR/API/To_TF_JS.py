import numpy as np

from AlignAIR.Data.PredictionDataset import PredictionDataset
from AlignAIR.Trainers import Trainer

model_weights_path = 'C:/Users/tomas/Downloads/AlignAIR_IGH_EXT_Model_weights_final_epoch'
model_type = 'heavy'

if model_type == 'heavy':
    from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
    dataconfig = None
    with open('C:/Users/tomas/Downloads/IGH_EXTENDED_DATACONFIG.pkl', 'rb') as f:
        from GenAIRR.dataconfig import DataConfig
        import pickle
        dataconfig = pickle.load(f)


    model_params = {'max_seq_length': 576, 'v_allele_count': dataconfig.number_of_v_alleles,
                    'd_allele_count': dataconfig.number_of_d_alleles+1,
                    'j_allele_count': dataconfig.number_of_j_alleles}
    model = HeavyChainAlignAIRR(**model_params)
elif model_type == 'tcrb':
    from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
    from GenAIRR.data import builtin_tcrb_data_config
    data_config = builtin_tcrb_data_config()
    print(dir(data_config))
    model_params = {'max_seq_length': 576,
                    'v_allele_count': data_config.number_of_v_alleles,
                    'd_allele_count': data_config.number_of_d_alleles+1,
                    'j_allele_count': data_config.number_of_j_alleles}
    model = HeavyChainAlignAIRR(**model_params)
else:
    from AlignAIR.Models.LightChain import LightChainAlignAIRR
    from GenAIRR.data import builtin_lambda_chain_data_config, builtin_kappa_chain_data_config
    kappa_dataconfig = builtin_kappa_chain_data_config()
    lambda_dataconfig = builtin_lambda_chain_data_config()

    kappa_v_alleles = {i.name:i for j in kappa_dataconfig.v_alleles for i in kappa_dataconfig.v_alleles[j]}
    kappa_j_alleles = {i.name:i for j in kappa_dataconfig.j_alleles for i in kappa_dataconfig.j_alleles[j]}

    lambda_v_alleles = {i.name:i for j in lambda_dataconfig.v_alleles for i in lambda_dataconfig.v_alleles[j]}
    lambda_j_alleles = {i.name:i for j in lambda_dataconfig.j_alleles for i in lambda_dataconfig.j_alleles[j]}

    v_count = len(kappa_v_alleles) + len(lambda_v_alleles)
    j_count = len(kappa_j_alleles) + len(lambda_j_alleles)
    model_params = {'max_seq_length': 576, 'v_allele_count': v_count, 'j_allele_count': j_count}
    model = LightChainAlignAIRR(**model_params)

trainer = Trainer(
    model=model,
    epochs=1,
    batch_size=32,
    steps_per_epoch=1,
    verbose=1,
)
MODEL_CHECKPOINT = model_weights_path
trainer.model.build({"tokenized_sequence": (None, model_params['max_seq_length'])})

trainer.load_model(MODEL_CHECKPOINT,max_seq_length=model_params['max_seq_length'])


prediction_Dataset = PredictionDataset(max_sequence_length=576)
seq = 'CAGCCACAACTGAACTGGTCAAGTCCAGGACTGGTGAATACCTCGCAGACCGTCACACTCACCCTTGCCGTGTCCGGGGACCGTGTCTCCAGAACCACTGCTGTTTGGAAGTGGAGGGGTCAGACCCCATCGCGAGGCCTTGCGTGGCTGGGAAGGACCTACNACAGTTCCAGGTGATTTGCTAACAACGAAGTGTCTGTGAATTGTTNAATATCCATGAACCCAGACGCATCCANGGAACGGNTCTTCCTGCACCTGAGGTCTGGGGCCTTCGACGACACGGCTGTACATNCGTGAGAAAGCGGTGACCTCTACTAGGATAGTGCTGAGTACGACTGGCATTACGCTCTCNGGGACCGTGCCACCCTTNTCACTGCCTCCTCGG'
es = prediction_Dataset.encode_and_equal_pad_sequence(seq)['tokenized_sequence']
predicted = trainer.model.predict({'tokenized_sequence':np.vstack([es])})

dummy_input = {
    "tokenized_sequence": np.zeros((1, model_params['max_seq_length']), dtype=np.float32),
}
_ = trainer.model(dummy_input)  # Build the model by invoking it

folder_path = 'C:/Users/tomas/Downloads/alignair-ihgv-extended-js'

trainer.model.save(folder_path)
# validate folder in saved path
import os
status = os.path.exists(folder_path)
print("Converted model saved in folder: ", status)