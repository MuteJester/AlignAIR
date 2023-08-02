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
model_path = (
    "/localdata/alignairr_data/sf5_unboundedadd_add_remove_coef_110/saved_models/tmp"
)
pred_file_path = "/localdata/alignairr_data/naive_repertoires/naive_sequences_clean.tsv"
# pred_file_path = (
#     "/localdata/alignairr_data/1M_for_test/sim_data_1M_asc_s5f_rate_001.tsv"
# )
saved_pred_dict_path = (
    "/localdata/alignairr_data/sf5_unboundedadd_add_remove_coef_110/pred/"
)
train_dataset_path = "/localdata/alignairr_data/sf5_unboundedadd_coef_110_v2/pred/"

if not os.path.exists(saved_pred_dict_path):
    os.makedirs(saved_pred_dict_path)

# Create a Trainer instance with desired parameters
trainer = Trainer(
    model=model,
    epochs=1,
    batch_size=batch_size,
    verbose=True,
    train_dataset=pred_file_path,
)


trainer.rebuild_model()
trainer.load_model(model_path)

pred_dict = trainer.predict(pred_file_path)

save_pred_dict_name = "model__" + model_name + "__data__" + "naive_sequences_clean.pkl"

save_path = os.path.join(saved_pred_dict_path, save_pred_dict_name)
print(save_path)
# Open a file and use dump()
with open(save_path, "wb") as file:
    # A new file will be created
    pickle.dump(pred_dict, file)
