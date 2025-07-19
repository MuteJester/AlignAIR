import os
import argparse
import pickle

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import logging
# Assuming the following modules are in your package structure
from AlignAIR.Data import SingleChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Models.LightChain import LightChainAlignAIRR
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
from AlignAIR.Trainers import Trainer


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlignAIRR Model")
    parser.add_argument("--genairr_dataconfig", required=True, help="a name of a builtin GenAIRR dataconfig or a path to a custom genairr dataconfig pkl file")
    parser.add_argument("--train_dataset", required=True, help="Path to the training dataset")
    parser.add_argument("--session_path", required=True, help="Base directory for session output")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--steps_per_epoch", type=int, default=512, help="number of samples to go through each epoch")
    parser.add_argument("--max_sequence_length", type=int, default=576, help="maximum input size")
    parser.add_argument("--model_name", default="AlignAIRR_Model", help="Name of the model for saving")
    args = parser.parse_args()
    return args

def get_dataconfig(dataconfig):
    import GenAIRR.data as data
    # check if the dataconfig is a path to a pkl file
    if os.path.exists(dataconfig):
        # load pickled dataconfig
        with open(dataconfig, 'rb') as f:
            data_config = pickle.load(f)
        return data_config
    elif dataconfig in data._CONFIG_NAMES:
        # check if the dataconfig is a builtin GenAIRR dataconfig
        data_config = getattr(data, dataconfig)
        return data_config
    else:
        # if the dataconfig is not a path or a builtin GenAIRR dataconfig, raise an error
        raise ValueError(f"Invalid dataconfig: {dataconfig}. It should be a path to a pkl file or a builtin GenAIRR dataconfig name.")


def main():
    setup_logging()
    args = parse_args()

    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    logging.info("Configuration setup is done.")
    logging.info(f"The Script will train a {args.genairr_dataconfig} AlignAIR")


    models_path = os.path.join(args.session_path, "saved_models")
    checkpoint_path = os.path.join(models_path, f"{args.model_name}_checkpoint")
    logs_path = args.session_path

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="v_allele_auc",
        factor=0.9,
        patience=20,
        min_delta=0.001,
        mode="auto",
    )

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
        mode="min",
    )

    logging.info("Callbacks are defined.")

    # Prepare dataset
    dataconfig = get_dataconfig(args.genairr_dataconfig)
    train_dataset = SingleChainDataset(data_path = args.train_dataset,dataconfig=dataconfig,batch_size=args.batch_size,
                                       max_sequence_length = args.max_sequence_length,use_streaming=True)

    logging.info("Dataset is loaded and prepared.")

    model_parmas = train_dataset.generate_model_params()

    model = SingleChainAlignAIR(**model_parmas)
    metrics = {
        'v_start': tf.keras.losses.mse,
        'v_end': tf.keras.losses.mse,
        'j_start': tf.keras.losses.mse,
        'j_end': tf.keras.losses.mse,
        'v_allele': tf.keras.losses.binary_crossentropy,
        'j_allele': tf.keras.losses.binary_crossentropy,
        }
    if dataconfig.metadata.has_d:
        metrics.update({
            'd_start': tf.keras.losses.mse,
            'd_end': tf.keras.losses.mse,
            'd_allele': tf.keras.losses.binary_crossentropy,
        })

    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1),
                  loss=None,
                    metrics=metrics
                  )

    logging.info("Model is Built")

    # Initialize the model and trainer
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=int(args.steps_per_epoch),
        verbose=1,
        classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
        regression_metric=tf.keras.losses.binary_crossentropy,
        log_to_file=True,
        log_file_name=args.model_name,
        log_file_path=logs_path,
        callbacks=[reduce_lr, model_checkpoint_callback],
    )

    logging.info("Starting model training.")
    trainer.train(train_dataset)

    # Save model and weights
    path_to_model_weights = os.path.join(models_path, args.model_name)
    os.makedirs(path_to_model_weights, exist_ok=True)

    trainer.model.save_weights(os.path.join(path_to_model_weights, f'{args.model_name}_weights_final_epoch'))
    logging.info("Model training is complete and model is saved.")


if __name__ == "__main__":
    main()
