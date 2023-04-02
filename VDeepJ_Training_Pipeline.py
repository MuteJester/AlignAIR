import pandas as pd
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm

from VDeepJDataset import VDeepJDataset
tqdm.pandas()
from VDeepJDataPrepper import VDeepJDataPrepper
from VDeepJModel import VDeepJAllign

# To-Do: Convert all the code below into a class that will handle the training given the dataset, similar to what transformer "Trainer" does.
# Step 1: Create VDeepJ Dataset
max_seq_length = 512
path = 'E:\igor_generated_5M\shm_rep_airship_0.8_flatvdj_gene.tsv'
batch_size = 64

VdeepJ_Dataset = VDeepJDataset(path,max_seq_length)


# Step 2: Get Train Dataset
train_dataset = VdeepJ_Dataset.get_train_generator(batch_size,corrupt_beginning=True)

# Step 3: Create a VDeepJ instance
vdeepj = VDeepJAllign(**VDeepJDataset.generate_model_params())

# Step 4: Compile the model instance
vdeepj.compile(optimizer=tf.optimizers.Adam(),#'adam',
                       loss=None,
                       metrics={'v_start':'mae','v_end':'mae','d_start':'mae','d_end':'mae','j_start':'mae','j_end':'mae',
                                'v_family':'categorical_accuracy',
                                'v_gene':'categorical_accuracy',
                                'v_allele':'categorical_accuracy',
                                'd_family':'categorical_accuracy',
                                'd_gene':'categorical_accuracy',
                                'd_allele':'categorical_accuracy',
                                'j_gene':'categorical_accuracy',
                                'j_allele':'categorical_accuracy',

                               })






# Step 10: Fit Model
steps_per_epoch = len(VdeepJ_Dataset) // batch_size
epochs = 2

history = vdeepj.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=0,
    callbacks=[TqdmCallback(1)]

)


