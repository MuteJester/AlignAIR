import os
from VDeepJDataset import VDeepJDatasetSingleBeamSegmentation
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from UnboundedTrainer import SingleBeamUnboundedTrainer
from Trainer import SingleBeamTrainer,SingleBeamSegmentationTrainer
from VDeepJModelExperimental import VDeepJAllignExperimentalSingleBeamConvSegmentationV2
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np

trainer = SingleBeamSegmentationTrainer(
    model=VDeepJAllignExperimentalSingleBeamConvSegmentationV2,
    data_path = "/localdata/alignairr_data/AlignAIRR_Large_Train_Dataset/AlignAIRR_Large_Train_Dataset.csv",
    batch_read_file=True,
    epochs=1,
    batch_size=1,
    steps_per_epoch=150_000,
    verbose=1,
    corrupt_beginning=True,
    classification_head_metric=[tf.keras.metrics.AUC(),tf.keras.metrics.AUC(),tf.keras.metrics.AUC()],
    interval_head_metric=tf.keras.losses.mae,
    corrupt_proba=0.7,
    airrship_mutation_rate=0.25,
    nucleotide_add_coef=210,
    nucleotide_remove_coef=330,
    random_sequence_add_proba=0.45,
    single_base_stream_proba=0.05,
    duplicate_leading_proba=0.25,
    random_allele_proba=0.25,
    num_parallel_calls=32,
)



trainer.model.build({'tokenized_sequence':(512,1)})
#trainer.model.load_weights('E:\\Immunobiology\\AlignAIRR\\\\sf5_unbounded_experimentalv4_model')
trainer.model.load_weights("/localdata/alignairr_data/sf5_alignairr_segmentation_no_regularization_residual_conn/saved_models/sf5_alignairr_segmentation_no_regularization_residual_conn")
print('Model Loaded!')


#print(trainer.model.v_start_mid.weights)


target = pd.read_table("/localdata/alignairr_data/naive_repertoires/naive_sequences_clean.tsv")
print('Target Dataset Loaded!')

eval_dataset_ = trainer.train_dataset.tokenize_sequences(target.sequence.to_list())
print('Train Dataset Encoded!')

padded_seqs_tensor = tf.convert_to_tensor(eval_dataset_, dtype=tf.int32)
dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
    'tokenized_sequence': padded_seqs_tensor})
dataset = (
    dataset_from_tensors
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)



raw_predictions = trainer.model.predict(dataset, verbose=True)

with open('/localdata/alignairr_data/sf5_alignairr_segmentation_no_regularization_residual_conn/'+'sf5_alignairr_segmentation_no_regularization_residual_conn_p1_p11_prediction_raw_14_10_2023.pkl','wb') as h:
    pickle.dump(raw_predictions,h)
    
print('Predicted And Saved!')

