# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from VDeepJModel import VDeepJAllign
from UnboundedTrainer import UnboundedTrainer
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.keras import WandbMetricsLogger
import numpy as np
import random
from itertools import product

seed = 42
np.random.seed(seed)  # Set the random seed
tf.random.set_seed(seed)  # Set the random seed
random.seed(seed)  # Set the random seed

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

### Start - CallBacks Definitions ###
class ChangeGeneMaskingCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_to_change):
        super(ChangeGeneMaskingCallback, self).__init__()
        self.epoch_to_change = epoch_to_change

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.epoch_to_change:
            if epoch == self.epoch_to_change:
                self.model.use_gene_masking = True
                print(f"Changed use_gene_masking to True at epoch {epoch+1}.")
### End - CallBacks Definitions ###

### Start - Parameters ###
patience = 3
epoch_to_change = 2  # Specify the epoch at which you want to add masking
epochs = 10
batch_size = 128
model_name = "s5f"
model = VDeepJAllign
noise_type = (
    "S5F_rate"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)

datasets_path = "/dsi/shared/ig/ig05_train"
session_name = "sf5_unbounded_hpo"
# pretrained_path = "/localdata/alignairr_data/models_2M_version23_sf5/saved_models/tmp"
session_path = os.path.join("/dsi/shared/ig/ig05_train", session_name)
models_path = os.path.join(session_path, "saved_models")
checkpoint_path = os.path.join(models_path, "tmp")
logs_path = os.path.join(session_path, "logs/")
### End - Parameters ###

### Start - HPO Grid ###
mutation_rate_grid = [0.01, 0.05, 0.08, 0.15, 0.2, 0.25]
steps_per_epoch_grid = [100_000, 350_000, 500_000, 850_000]
### End - HPO Grid ###

if not os.path.exists(session_path):
    os.makedirs(session_path)
    os.makedirs(models_path)
    os.makedirs(logs_path)
    os.makedirs(checkpoint_path)

for mutation_rate, steps_per_epoch in product(mutation_rate_grid, steps_per_epoch_grid):

    str_mutation_rate = str(mutation_rate).replace(".", "")
    model_name = f"s5f__noise_mag_{str_mutation_rate}__steps_per_epoch_{steps_per_epoch}"

    ### Start - wandb ###
    run = wandb.init(
        project=session_name, name=model_name, settings=wandb.Settings(code_dir="."), group=str_mutation_rate
    )
    # Hyperparameters
    config = wandb.config
    config.model_name = model_name
    config.session_name = session_name
    config.epochs = epochs
    config.batch_size = batch_size
    config.patience = patience
    ### Start - wandb ###

    ### Start - Create Keras Callback Objects ###
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
        mode="min",
    )
    wandb_callback = WandbMetricsLogger(log_freq="batch")
    change_gene_masking_callback = ChangeGeneMaskingCallback(epoch_to_change)
    early_stopping = EarlyStopping(
        monitor="loss", patience=patience, min_delta=0.005, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="v_allele_categorical_accuracy",
        factor=0.9,
        patience=1,
        min_delta=0.0001,
        mode="max",
    )
    ### End - Create Keras Callback Objects ###

    trainer = UnboundedTrainer(
        VDeepJAllign,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        corrupt_beginning=True,
        classification_head_metric="categorical_accuracy",
        interval_head_metric="mae",
        corrupt_proba=0.5,
        use_gene_masking=False,
        log_to_file=True,
        log_file_name=model_name,
        log_file_path=logs_path,
        # pretrained=pretrained_path,
        callbacks=[
            reduce_lr,
            change_gene_masking_callback,
            early_stopping,
            wandb_callback,
            model_checkpoint_callback,
        ],
        optimizers_params={"clipnorm": 1},
        mutation_rate=mutation_rate,
    )

    # Train the model
    trainer.train()

    path_to_model_weights = os.path.join(models_path, model_name)
    os.mkdir(path_to_model_weights)

    trainer.save_model(path_to_model_weights)

    run.finish()
