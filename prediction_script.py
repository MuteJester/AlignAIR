# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from VDeepJModel import VDeepJAllign
import os
import pickle
from tqdm import tqdm


model = VDeepJAllign
batch_size = 512

datasets_path = "/localdata/alignairr_data/1M_for_test/"
models_path = "/localdata/alignairr_data/models_2M_version19/saved_models/"
saved_pred_dict_path = "/localdata/alignairr_data/models_2M_version19/pred/"
noise_type = (
    "S5F_rate"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)

for file in tqdm(os.listdir(models_path), desc="model"):
    if ".data-" in file and "add_n" in file:
        model_name = file.split("VDeepJ")[0]

        if noise_type == "S5F_rate":
            train_dataset_path = os.path.join(
                datasets_path,
                model_name.replace("2M", "1M").replace("_HH", "").replace("S5F", "s5f")
                + ".tsv",
            )
        else:
            train_dataset_path = os.path.join(
                datasets_path, model_name.replace("2M", "1M") + ".tsv"
            )

        # Create a Trainer instance with desired parameters
        trainer = Trainer(
            model=model,
            epochs=1,
            batch_size=batch_size,
            train_dataset=train_dataset_path,
            verbose=True,
        )

        model_path = os.path.join(models_path, file.split(".data-")[0])

        trainer.rebuild_model()
        trainer.load_model(model_path)

        for pred_file in tqdm(os.listdir(datasets_path), desc="data"):
            if pred_file.endswith(".tsv") and "add_n" in pred_file:
                pred_file_path = os.path.join(datasets_path, pred_file)

                pred_dict = trainer.predict(pred_file_path)

                save_pred_dict_name = (
                    "model__"
                    + model_name
                    + "__data__"
                    + pred_file.split(".")[0]
                    + ".pkl"
                )

                save_path = os.path.join(saved_pred_dict_path, save_pred_dict_name)

                # Open a file and use dump()
                with open(save_path, "wb") as file:
                    # A new file will be created
                    pickle.dump(pred_dict, file)
