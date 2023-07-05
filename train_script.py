# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from VDeepJModel import VDeepJAllign
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau

Reduce_lr = ReduceLROnPlateau(
    monitor="v_allele_categorical_accuracy",
    factor=0.8,
    patience=1,
    min_delta=1e-32,
    mode="min",
)


class ChangeParameterCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_to_change):
        super(ChangeParameterCallback, self).__init__()
        self.epoch_to_change = epoch_to_change

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.epoch_to_change:
            if epoch == self.epoch_to_change:
                self.model.use_gene_masking = True
                print(f"Changed use_gene_masking to True at epoch {epoch+1}.")


epoch_to_change = 2  # Specify the epoch at which you want to change the parameter
change_parameter_callback = ChangeParameterCallback(epoch_to_change)

# Define your model
model = VDeepJAllign
# Define other parameters
epochs = 5
batch_size = 64
datasets_path = "/dsi/shared/ig/ig05_train"
session_name = "models_2M_s5f_20"
session_path = os.path.join("/dsi/shared/ig/ig05_models/", session_name)
models_path = os.path.join(session_path, "saved_models/")
logs_path = os.path.join(session_path, "logs/")

if not os.path.exists(session_path):
    os.makedirs(session_path)
    os.makedirs(models_path)
    os.makedirs(logs_path)


for file in os.listdir(datasets_path):
    if file.endswith(".tsv"):
        # For Debug ###################
        if "s5f_20" in file and file.endswith(".tsv"):
            train_dataset_path = os.path.join(datasets_path, file)
            model_name = file.split(".")[0]

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
                callbacks=[change_parameter_callback],
                use_gene_masking=False,
                # For Debug ###################
                # optimizers_params={"clipnorm": 1},
            )

            # Train the model
            trainer.train()

            path_to_model_weights = os.path.join(models_path, model_name)
            os.mkdir(path_to_model_weights)

            trainer.save_model(path_to_model_weights)
