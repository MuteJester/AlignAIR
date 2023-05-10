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
# train_path = "/home/bcrlab/eisenbr2/old_vdj_nlp/vdj_nlp/airrship_data/train/airrship_01_train.tsv"
train_path = (
    "/home/bcrlab/eisenbr2/old_vdj_nlp/vdj_nlp/airrship_data/val/airrship_01_val.tsv"
)
val_path = (
    "/home/bcrlab/eisenbr2/old_vdj_nlp/vdj_nlp/airrship_data/val/airrship_01_val.tsv"
)

# train_path = "/localdata/alignairr_data/10M/sim_data_10M_asc.tsv"
# val_path = "/localdata/alignairr_data/1M/sim_data_1M_asc.tsv"

batch_size = 64
train_mode = 1
eval_mode = 0
epochs = 1


train_vdeepJ_Dataset = VDeepJDataset(train_path, max_seq_length)
val_vdeepJ_Dataset = VDeepJDataset(val_path, max_seq_length)


# Step 2: Get Train Dataset
train_dataset = train_vdeepJ_Dataset.get_train_generator(
    batch_size, corrupt_beginning=True
)
val_dataset = val_vdeepJ_Dataset.get_train_generator(batch_size, corrupt_beginning=True)

# Step 3: Create a VDeepJ instance
vdeepj = VDeepJAllign(**train_vdeepJ_Dataset.generate_model_params())

# Step 4: Compile the model instance
vdeepj.compile(
    optimizer=tf.optimizers.Adam(),  #'adam',
    loss=None,
    metrics={
        "v_start": "mae",
        "v_end": "mae",
        "d_start": "mae",
        "d_end": "mae",
        "j_start": "mae",
        "j_end": "mae",
        "v_family": "categorical_accuracy",
        "v_gene": "categorical_accuracy",
        "v_allele": "categorical_accuracy",
        "d_family": "categorical_accuracy",
        "d_gene": "categorical_accuracy",
        "d_allele": "categorical_accuracy",
        "j_gene": "categorical_accuracy",
        "j_allele": "categorical_accuracy",
    },
)

if eval_mode:
    x, y = next(iter(train_dataset))
    vdeepj.train_step([x, y])

if train_mode:
    # Step 10: Fit Model
    steps_per_epoch_train = len(train_vdeepJ_Dataset) // batch_size

    history = vdeepj.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch_train,
        verbose=0,
        callbacks=[TqdmCallback(1)],
    )

    # Step 11: Evauate the model
    steps_per_epoch_val = len(val_vdeepJ_Dataset) // batch_size

    score = vdeepj.evaluate(
        val_dataset, verbose=0, steps=steps_per_epoch_val, return_dict=True
    )
    print(score)
