# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from UnboundedTrainer import UnboundedTrainer
from VDeepJModel import VDeepJAllign
import os
import pickle
from tqdm import tqdm


model = VDeepJAllign
batch_size = 512

datasets_path = "/localdata/alignairr_data/1M_for_test/"
models_path = "/localdata/alignairr_data/models_2M_version19/saved_models/"

noise_type = (
    "S5F_rate"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)


model_name = "s5f_unbounded"
model_path = "/localdata/alignairr_data/models_2M_version22_sf5/saved_models/tmp"
pred_file_path = "/localdata/alignairr_data/naive_repertoires/naive_sequences_clean_2.tsv"
saved_pred_dict_path = "/localdata/alignairr_data/models_2M_version22_sf5/pred/"
train_dataset_path = (
    "/localdata/alignairr_data/naive_repertoires/naive_sequences_clean.tsv"
)

# Create a Trainer instance with desired parameters
trainer = Trainer(
    model=model,
    epochs=1,
    batch_size=batch_size,
    verbose=True,
    train_dataset=train_dataset_path,
)


trainer.rebuild_model()
trainer.load_model(model_path)

pred_dict = trainer.predict(pred_file_path)

save_pred_dict_name = "model__" + model_name + "__data__" + "naive_sequences_clean_2.pkl"

save_path = os.path.join(saved_pred_dict_path, save_pred_dict_name)

# Open a file and use dump()
with open(save_path, "wb") as file:
    # A new file will be created
    pickle.dump(pred_dict, file)
