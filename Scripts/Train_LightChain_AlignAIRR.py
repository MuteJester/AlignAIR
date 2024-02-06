import sys

# Let's say your module is in '/path/to/your/module'
module_dir = '/home/bcrlab/thomas/AlignAIRR/'

# Append this directory to sys.path
if module_dir not in sys.path:
    sys.path.append(module_dir)
    
# Import the necessary libraries
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
import wandb
from wandb.keras import WandbMetricsLogger
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import average_precision_score
import pandas as pd
from tensorflow_addons.optimizers import CyclicalLearningRate
from Models.LightChain import LightChainAlignAIRR
from Data import HeavyChainDataset,LightChainDataset
from SequenceSimulation.sequence import LightChainSequence
from VDeepJLayers import AlignAIRREvaluationCallback
import pickle
from Models.HeavyChain import HeavyChainAlignAIRR
from Trainers import Trainer
with open('/home/bcrlab/thomas/AlignAIRR/SequenceSimulation/data/LightChain_LAMBDA_DataConfig.pkl','rb') as h:
    lightchain_lambda_config = pickle.load(h)
with open('/home/bcrlab/thomas/AlignAIRR/SequenceSimulation/data/LightChain_KAPPA_DataConfig.pkl','rb') as h:
    lightchain_kappa_config = pickle.load(h)
    
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



# Define other parameters
epochs = 300
batch_size = 512
noise_type = (
    "S5F_rate"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)


model_name = "AlignAIRR_S5F_LightChain"
datasets_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/"
session_name = "S5F_AlignAIRR_LightChain"
session_path = os.path.join("/localdata/alignairr_data/", model_name)
models_path = os.path.join(session_path, "saved_models")
checkpoint_path = os.path.join(models_path, model_name)
logs_path = os.path.join(session_path, "logs/")
eval_cps_path = os.path.join(session_path, "evaluation_checkpoints/")

if not os.path.exists(session_path):
    os.makedirs(session_path)
    os.makedirs(models_path)
    os.makedirs(logs_path)
    os.makedirs(checkpoint_path)
    os.makedirs(eval_cps_path)

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



TRAIN_DATASET = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/LightChain_OGRDB_DataConfig_AlignAIRR_S5F_15M_with_Corruption_Mrate_003__025.csv"
train_dataset = LightChainDataset(data_path=TRAIN_DATASET
                                  ,lambda_dataconfig=lightchain_lambda_config,
                                  kappa_dataconfig = lightchain_kappa_config,
                                  batch_size=batch_size,
                                  max_sequence_length=512,
                                  batch_read_file=True)





print('Starting The Training...')
print('Train Dataset Path: ',TRAIN_DATASET) 

trainer = Trainer(
    model= LightChainAlignAIRR,
    dataset=train_dataset,
    epochs=epochs,
    steps_per_epoch=150_000,
    verbose=1,
    classification_metric=[tf.keras.metrics.AUC(),tf.keras.metrics.AUC(),tf.keras.metrics.AUC()],
    regression_metric=tf.keras.losses.binary_crossentropy,
    log_to_file=True,
    log_file_name=model_name,
    log_file_path=logs_path,
    callbacks=[
        reduce_lr,
        wandb_callback,
        model_checkpoint_callback,
    ],
    optimizers_params={"clipnorm": 1},
)

# Train the model
trainer.train()

path_to_model_weights = os.path.join(models_path, model_name)
os.mkdir(path_to_model_weights)

trainer.save_model(path_to_model_weights)

run.finish()
