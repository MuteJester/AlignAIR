# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from VDeepJModel import VDeepJAllign
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.keras import WandbMetricsLogger


### Start - CallBacks Definitions ###

Reduce_lr = ReduceLROnPlateau(
    monitor="v_allele_categorical_accuracy",
    factor=0.8,
    patience=1,
    min_delta=1e-32,
    mode="min",
)


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

patience = 5
epoch_to_change = 2  # Specify the epoch at which you want to change the parameter

change_gene_masking_callback = ChangeGeneMaskingCallback(epoch_to_change)
early_stopping = EarlyStopping(
    monitor="loss", patience=patience, restore_best_weights=True
)

# Define your model
model = VDeepJAllign

# Define other parameters
epochs = 20
batch_size = 64
noise_type = (
    "s5f_rate"  # Can be of types: ["s5f_rate", "s5f_20", "s5f_opposite", "uniform"]
)


datasets_path = "/localdata/alignairr_data/2M_for_training/"
session_name = "models_2M_version16"
session_path = os.path.join("/localdata/alignairr_data/", session_name)
models_path = os.path.join(session_path, "saved_models")
logs_path = os.path.join(session_path, "logs/")

if not os.path.exists(session_path):
    os.makedirs(session_path)
    os.makedirs(models_path)
    os.makedirs(logs_path)


for file in os.listdir(datasets_path):
    if file.endswith(".tsv"):
        # For Debug ###################
        if (noise_type in file) and (file.endswith(".tsv")) and ("add_n" in file):
            train_dataset_path = os.path.join(datasets_path, file)
            model_name = file.split(".")[0]
            noise_rate = model_name.split("_add_n")[0].split("rate_")[1]

            # Initialize wandb
            run = wandb.init(project=session_name, name=model_name)
            wandb_callback = WandbMetricsLogger(log_freq="batch")
            ### Hyperparameters
            config = wandb.config
            config.model_name = model_name
            config.session_name = session_name
            config.epochs = epochs
            config.batch_size = batch_size
            config.noise_type = noise_type if noise_type != "s5f_rate" else "s5f"
            config.noise_rate = noise_rate
            config.patience = patience

            # Create a Trainer instance with desired parameters
            trainer = Trainer(
                model=model,
                epochs=epochs,
                batch_size=batch_size,
                train_dataset=train_dataset_path,
                verbose=True,
                log_to_file=True,
                log_file_name=model_name,
                log_file_path=logs_path,
                callbacks=[
                    change_gene_masking_callback,
                    early_stopping,
                    wandb_callback,
                ],
                use_gene_masking=False,
                # For Debug ###################
                # optimizers_params={"clipnorm": 1},
            )

            # Train the model
            trainer.train()

            path_to_model_weights = os.path.join(models_path, model_name)
            os.mkdir(path_to_model_weights)

            trainer.save_model(path_to_model_weights)

            # DEBUG - Only one run
            # break

            run.finish()
