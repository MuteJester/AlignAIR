# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from VDeepJModel import VDeepJAllign
import os
import pickle
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model = VDeepJAllign
batch_size = 512

datasets_path = "/dsi/shared/ig/ig05_test/"
models_path = "/dsi/shared/ig/models_2M_version19/saved_models/"
saved_pred_dict_path = "/dsi/shared/ig/models_2M_version19/pred/"
noise_type = (
    "s5f_opposite"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)

for file in tqdm(os.listdir(models_path), desc="model"):
    if ".data-" in file and "add_n" in file and noise_type in file:
        model_name = file.split("VDeepJ")[0]
        train_dataset_path = os.path.join(
            datasets_path, model_name.replace("2M", "1M") + ".tsv"
        )

        if "asc" in train_dataset_path and noise_type == "S5F_20":
            train_dataset_path = train_dataset_path.replace("_asc", "")

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
            if (
                pred_file.endswith(".tsv")
                and "add_n" in pred_file
                and "new_param" not in pred_file
            ):
                pred_file_path = os.path.join(datasets_path, pred_file)

                save_pred_dict_name = (
                    "model__"
                    + model_name
                    + "__data__"
                    + pred_file.split(".")[0]
                    + ".pkl"
                )

                save_path = os.path.join(saved_pred_dict_path, save_pred_dict_name)

                if not os.path.exists(save_path):
                    pred_dict = trainer.predict(pred_file_path)

                    # Open a file and use dump()
                    with open(save_path, "wb") as file:
                        # A new file will be created
                        pickle.dump(pred_dict, file)
