from VDeepJModelExperimental import VDeepJAllignExperimentalV3,VDeepJAllignExperimentalSingleBeamRG
from UnboundedTrainer import SingleBeamUnboundedTrainer
from Trainer import Trainer
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import os
import copy
import pickle

model_path = r"/localdata/alignairr_data/sf5_alignairr_latest_mh_single_beam_RG_end_corrected/saved_models/sf5_alignairr_latest_mh_single_beam_RG_end_corrected_model"
saved_pred_dict_path = r"/localdata/alignairr_data/sf5_alignairr_latest_mh_single_beam_RG_end_corrected_pred/p1_p11_added_noise/"
noise_types = ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
datasets_path = r"/localdata/alignairr_data/p1_p11_added_noise/"
# datasets_path = r"/localdata/alignairr_data/1M_for_test/"
model_name = r"sf5_alignairr_latest_mh_single_beam_RG_end_corrected"

if not os.path.exists(saved_pred_dict_path):
   os.makedirs(saved_pred_dict_path)

trainer = SingleBeamUnboundedTrainer(
    VDeepJAllignExperimentalSingleBeamRG,
    epochs=10,
    batch_size=512,
    steps_per_epoch=150_000,
    verbose=1,
    corrupt_beginning=True,
    classification_head_metric=[tf.keras.metrics.AUC(),tf.keras.metrics.AUC(),tf.keras.metrics.AUC()],
    interval_head_metric=tf.keras.losses.mae,
    corrupt_proba=0.7,
    nucleotide_add_coef=210,
    nucleotide_remove_coef=330, 
    random_sequence_add_proba=0.45,
    single_base_stream_proba=0.05,
    duplicate_leading_proba=0.25,
    random_allele_proba=0.25,
    num_parallel_calls=32,
    log_to_file=True,
 
)
trainer.model.build({'tokenized_sequence':(512,1),'tokenized_sequence_for_masking':(512,1)})
trainer.model.load_weights(model_path)


for eval_dataset in tqdm(os.listdir(datasets_path), desc="data"):
    if eval_dataset.endswith(".tsv") and "add_n" in eval_dataset:
    # if eval_dataset.endswith(".tsv"):

        pred_file_name = copy.deepcopy(eval_dataset)
        eval_dataset_path = os.path.join(datasets_path, eval_dataset)

        dataset = pd.read_table(eval_dataset_path)
        eval_dataset = dataset['sequence']
        eval_dataset = eval_dataset.str.replace('-','')

        eval_dataset_ = trainer.train_dataset.tokenize_sequences(eval_dataset)
        padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.int32)
        dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
            'tokenized_sequence': padded_seqs_tensor,
            'tokenized_sequence_for_masking': padded_seqs_tensor
        })
        dataset = (
            dataset_from_tensors
            .batch(512)
            .prefetch(tf.data.AUTOTUNE)
        )

        pred_dict = trainer.model.predict(dataset, verbose=True)

        save_pred_dict_name = (
            "model__"
            + model_name
            + "__data__"
            + pred_file_name.split(".")[0]
            + ".pkl"
        )

        save_path = os.path.join(saved_pred_dict_path, save_pred_dict_name)

        # Open a file and use dump()
        with open(save_path, "wb") as file:
            # A new file will be created
            pickle.dump(pred_dict, file)