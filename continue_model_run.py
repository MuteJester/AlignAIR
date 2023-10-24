# Import the necessary libraries
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
import wandb
from wandb.keras import WandbMetricsLogger
import numpy as np
import random
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamRG
from Trainer import SingleBeamTrainer
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamRG
import tensorflow_addons as tfa

seed = 42
np.random.seed(seed)  # Set the random seed
tf.random.set_seed(seed)  # Set the random seed
random.seed(seed)  # Set the random seed

### Start - CallBacks Definitions ###

reduce_lr = ReduceLROnPlateau(
    monitor="v_allele_auc",
    factor=0.8,
    patience=10,
    min_delta=0.001,
    mode="auto",
)



patience = 5
epoch_to_change = 2  # Specify the epoch at which you want to change the parameter


# early_stopping = EarlyStopping(
#     monitor="v_start_log_cosh",
#     patience=patience,
#     min_delta=0.005,
#     restore_best_weights=True,
# )

# Define other parameters
epochs = 1500
batch_size = 128
noise_type = (
    "S5F_rate"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)


datasets_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/"
session_name = "sf5_alignairr_latest"
session_path = os.path.join("/localdata/alignairr_data/", session_name+'_mh_single_beam_RG_end_corrected_SGD_CONTINUED_LAMB')
models_path = os.path.join(session_path, "saved_models")
checkpoint_path = os.path.join(models_path, "sf5_alignairr_latest_mh_single_beam_RG_end_corrected_model_SGD_CONTINUED_LAMB")
logs_path = os.path.join(session_path, "logs/")

if not os.path.exists(session_path):
    os.makedirs(session_path)
    os.makedirs(models_path)
    os.makedirs(logs_path)
    os.makedirs(checkpoint_path)

model_name = "sf5_alignairr_latest"
# Initialize wandb
run = wandb.init(
    project=session_name,
    name=model_name,
    settings=wandb.Settings(code_dir="."),
    entity="thomaskon90",
)

wandb_callback = WandbMetricsLogger(log_freq="batch")
### Hyperparameters
config = wandb.config
config.model_name = model_name
config.session_name = session_name
config.epochs = epochs
config.batch_size = batch_size
config.patience = patience

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor="loss",
    mode="min",
)


print('Starting The Training...')
trainer = SingleBeamTrainer(
    model=VDeepJAllignExperimentalSingleBeamRG,
    data_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/AlignAIRR_Large_Train_Dataset.csv",
    batch_read_file=True,
    epochs=epochs,
    batch_size=batch_size,
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
    log_to_file=True,
    log_file_name=model_name,
    log_file_path=logs_path,
    allele_map_path='',
    #pretrained=pretrained_path,
    callbacks=[
        reduce_lr,
        # early_stopping,
        wandb_callback,
        model_checkpoint_callback,
    ],
    optimizers_params={"clipnorm": 1},#,'min_lr':1e-8},
    optimizers=tfa.optimizers.LAMB#tfa.optimizers.AdaBelief#tf.keras.optimizers.SGD
)

trainer.model.build({'tokenized_sequence':(512,1),'tokenized_sequence_for_masking':(512,1)})
trainer.model.load_weights('/localdata/alignairr_data/sf5_alignairr_latest_mh_single_beam_RG_end_corrected_SGD/saved_models/sf5_alignairr_latest_mh_single_beam_RG_end_corrected_model_SGD')
print(trainer.model.weights)

# from tensorflow.keras import backend as K
# K.set_value(trainer.model.optimizer.learning_rate, 0.001)


# Train the model
trainer.train()

path_to_model_weights = os.path.join(models_path, model_name)
os.mkdir(path_to_model_weights)

trainer.save_model(path_to_model_weights)

run.finish()

