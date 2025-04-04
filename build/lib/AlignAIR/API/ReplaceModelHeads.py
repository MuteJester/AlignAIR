import pandas as pd
import argparse
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import random
import tensorflow as tf
import pandas as pd
from AlignAIR.Data import HeavyChainDataset,LightChainDataset
from GenAIRR.sequence import LightChainSequence
import pickle
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR
from AlignAIR.Trainers import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlingAIRR Model Trainer with Head Replacement')
    parser.add_argument('--new_dataconfig_path', type=str, required=True,
                        help='path to new dataconfig with new alleles')
    parser.add_argument('--chain_type', type=str, required=True,
                        help='heavy/light')
    parser.add_argument('--pretrained_model_weights', type=str, required=True,
                        help='path to pretraind AlignAIRR weights file')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path_to_save_new_model')
    args = parser.parse_args()

    trainer = Trainer(
        model=LightChainAlignAIRR,
        dataset=lcd,
        epochs=1,
        steps_per_epoch=150_000,
        verbose=1,
        classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
        regression_metric=tf.keras.losses.binary_crossentropy,

    )
    trainer.model.build({'tokenized_sequence': (512, 1)})

    MODEL_CHECKPOINT = r"C:\Users\Tomas\Downloads\AlignAIRR_Refactored_S5F_LightChain\saved_models\AlignAIRR_Refactored_S5F_LightChain"
    print('Loading: ', MODEL_CHECKPOINT.split('/')[-1])
    trainer.model.load_weights(
        MODEL_CHECKPOINT)
    print('Model Loaded!')

    # modify heads
    #trainer.model.v_allele_call_head = Dense()
    #trainer.model.d_allele_call_head
    #trainer.model.j_allele_call_head

    # freeze all layers except new ones
    for layer in trainer.model.layers:
        if layer.name not in ['v_allele','d_allele','j_allele']:
            layer.trainable = False
