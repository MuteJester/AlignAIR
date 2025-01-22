import numpy as np

from AlignAIR.Data.PredictionDataset import PredictionDataset
from AlignAIR.Trainers import Trainer

model_weights_path = 'C:/Users/tomas/Desktop/AlignAIRR/tests/AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2'

from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR

model_params = {'max_seq_length': 576, 'v_allele_count': 198, 'd_allele_count': 34, 'j_allele_count': 7}
model = HeavyChainAlignAIRR(**model_params)
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


trainer.model.save('C:/Users/tomas/Downloads/latest_alignair_model')
