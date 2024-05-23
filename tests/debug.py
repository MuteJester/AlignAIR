import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

dsf = Path(r'C:\Users\tomas\Desktop\AlignAIRR\tests\sample_HeavyChain_dataset.csv')
from GenAIRR.data import builtin_heavy_chain_data_config
from src.AlignAIR.Data import HeavyChainDataset
from src.AlignAIR.Trainers import Trainer
from src.AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
import pandas as pd
train_dataset = HeavyChainDataset(data_path=dsf
                                      , dataconfig=builtin_heavy_chain_data_config(),
                                                      batch_size=32,
                                                      max_sequence_length=576,
                                              batch_read_file=True)

trainer = Trainer(
    model=HeavyChainAlignAIRR,
    dataset=train_dataset,
    epochs=1,
    steps_per_epoch=1,
    verbose=1,
)
trainer.model.build({'tokenized_sequence': (576, 1)})

MODEL_CHECKPOINT = r'C:\Users\tomas\Desktop\AlignAIRR\tests\AlignAIRR_S5F_OGRDB_Experimental_New_Loss_V7'
trainer.model.load_weights(MODEL_CHECKPOINT)

seqs = pd.read_csv(dsf)
encoded,pads  = trainer.train_dataset.encode_and_pad_sequences(seqs.sequence)
print(encoded)
pred = trainer.model.predict({'tokenized_sequence':np.vstack(encoded)})
with open(r'C:\Users\tomas\Desktop\temp.pkl','wb') as h:
    import pickle
    pickle.dump((pred,pads,seqs),h)