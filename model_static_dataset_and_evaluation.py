# Import the necessary libraries
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
import wandb
from wandb.keras import WandbMetricsLogger
import numpy as np
import random
from VDeepJDataset import VDeepJDatasetSingleBeamSegmentation
from VDeepJLayers import RealDataEvaluationCallback
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRF_HP, \
    VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRF, \
    VDeepJAllignExperimentalSingleBeamConvSegmentationResidualIndel
from Trainer import SingleBeamSegmentationTrainer, SingleBeamTrainer, SingleBeamSegmentationTrainerV2
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamRG
import tensorflow as tf
from sklearn.metrics import average_precision_score
import pandas as pd

def average_precision(y_true, y_pred):
    # Calculate AP for each class
    ap_per_class = []
    for i in range(y_true.shape[1]):
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        ap_per_class.append(ap)
    
    # Return the mean AP across all classes
    return tf.reduce_mean(ap_per_class)

seed = 42
np.random.seed(seed)  # Set the random seed
tf.random.set_seed(seed)  # Set the random seed
random.seed(seed)  # Set the random seed

### Start - CallBacks Definitions ###

reduce_lr = ReduceLROnPlateau(
    monitor="v_allele_auc",
    factor=0.9,
    patience=20,
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
epochs = 2500
batch_size = 512
noise_type = (
    "S5F_rate"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)


datasets_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/"
session_name = "sf5_alignairr_segmentation"
session_path = os.path.join("/localdata/alignairr_data/", session_name+'_residual_w_indels')
models_path = os.path.join(session_path, "saved_models")
checkpoint_path = os.path.join(models_path, "sf5_alignairr_segmentation_residual_w_indels")
logs_path = os.path.join(session_path, "logs/")
eval_cps_path = os.path.join(session_path, "evaluation_checkpoints/")

if not os.path.exists(session_path):
    os.makedirs(session_path)
    os.makedirs(models_path)
    os.makedirs(logs_path)
    os.makedirs(checkpoint_path)
    os.makedirs(eval_cps_path)

model_name = "sf5_alignairr_segmentation_residual_w_indels"
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


p1_p11_data = pd.read_table("/localdata/alignairr_data/naive_repertoires/naive_sequences_clean.tsv",usecols=['sequence','v_call'])
print('P1 P11 Dataset Loaded!...')

train_dataset = VDeepJDatasetSingleBeamSegmentation(
            data_path="/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/AlignAIRR_Large_Train_Dataset_with_MUTRATE.csv",
            max_sequence_length=512,
            corrupt_beginning=True,
            corrupt_proba=0.7,
            nucleotide_add_coef=33,
            nucleotide_remove_coef=33,
            batch_size=1,
            randomize_rate=False,
            mutation_rate=0.25,
            random_sequence_add_proba=0.65,
            single_base_stream_proba=0.05,
            duplicate_leading_proba=0.15,
            random_allele_proba=0.15,
            batch_read_file=True
        )

p1p11_evaluation_callback = RealDataEvaluationCallback(validation_data=(p1_p11_data.sequence.to_list(), p1_p11_data.v_call.to_list()),train_dataset=train_dataset,
                                                       filepath=eval_cps_path,
                                                       period=10)
print('Evaluation Callback Created!...')


print('Starting The Training...')
trainer = SingleBeamSegmentationTrainerV2(
    model= VDeepJAllignExperimentalSingleBeamConvSegmentationResidualIndel,
    data_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/AlignAIRR_Large_Train_Dataset_with_MUTRATE.csv",
    batch_read_file=True,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=150_000,
    verbose=1,
    corrupt_beginning=True,
    classification_head_metric=[tf.keras.metrics.AUC(),tf.keras.metrics.AUC(),tf.keras.metrics.AUC()],
    interval_head_metric=tf.keras.losses.binary_crossentropy,
    corrupt_proba=0.7,
    airrship_mutation_rate=0.25,
    nucleotide_add_coef=210,
    nucleotide_remove_coef=320,
    random_sequence_add_proba=0.65,
    single_base_stream_proba=0.05,
    duplicate_leading_proba=0.15,
    random_allele_proba=0.15,
    insertion_proba=0.5,
    deletions_proba=0.5,
    deletion_coef=10,
    insertion_coef=10,
    num_parallel_calls=32,
    log_to_file=True,
    log_file_name=model_name,
    log_file_path=logs_path,
    allele_map_path='',
    callbacks=[
        reduce_lr,
        p1p11_evaluation_callback,
        wandb_callback,
        model_checkpoint_callback,
    ],
    optimizers_params={"clipnorm": 1}

)

# Train the model
trainer.train()

path_to_model_weights = os.path.join(models_path, model_name)
os.mkdir(path_to_model_weights)

trainer.save_model(path_to_model_weights)

run.finish()
