# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from VDeepJModel import VDeepJAllign
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
import time

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[1], True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# if 0:
#     # from tensorflow.compat.v1 import ConfigProto
#     # from tensorflow.compat.v1 import InteractiveSession

#     # def fix_gpu():
#     #     config = ConfigProto()
#     #     config.gpu_options.allow_growth = True
#     #     session = InteractiveSession(config=config)

#     # fix_gpu()
#     # s = 4

#     physical_devices = tf.config.experimental.list_physical_devices("GPU")
#     # physical_devices = tf.config.experimental.list_physical_devices('CPU')
#     print("physical_devices-------------", len(physical_devices))
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

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


start = time.time()
# Define your model
model = VDeepJAllign
# Define other parameters
epochs = 5
batch_size = 64
# specific_dataset = "sim_data_1M_S5F_20_rate_001_add_n.tsv"
# datasets_path = "/dsi/shared/ig/ig05_test/"  # Only change this
# train_dataset_path = os.path.join(datasets_path, specific_dataset)
train_dataset_path = os.path.join(r"sim_data_1M_S5F_20_rate_001_add_n.tsv")

trainer = Trainer(
    model=model,
    epochs=epochs,
    batch_size=batch_size,
    train_dataset=train_dataset_path,
    verbose=False,
    callbacks=[change_parameter_callback],
    use_gene_masking=False,
)

# Train the model
trainer.train()
end = time.time()

print("Time = ", end - start)
