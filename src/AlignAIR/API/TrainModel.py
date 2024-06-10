import os
import argparse
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import logging
# Assuming the following modules are in your package structure
from AlignAIR.Data import HeavyChainDataset,LightChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Trainers import Trainer
from GenAIRR.data import builtin_heavy_chain_data_config,builtin_kappa_chain_data_config,builtin_lambda_chain_data_config


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlignAIRR Model")
    parser.add_argument("--chain_type", required=True, help="heavy/light")
    parser.add_argument("--train_dataset", required=True, help="Path to the training dataset")
    parser.add_argument("--session_path", required=True, help="Base directory for session output")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--steps_per_epoch", type=int, default=512, help="number of samples to go through each epoch")
    parser.add_argument("--max_sequence_length", type=int, default=576, help="maximum input size")
    parser.add_argument("--model_name", default="AlignAIRR_Model", help="Name of the model for saving")
    args = parser.parse_args()
    return args

def get_dataconfig(chain_type):
    if chain_type == 'heavy':
        return builtin_heavy_chain_data_config()
    elif chain_type == 'light':
        return builtin_kappa_chain_data_config(),builtin_lambda_chain_data_config()

def get_train_dataset(chain_type, data_path,batch_size,max_sequence_length):
    data_config = get_dataconfig(chain_type)
    train_dataset = None

    if chain_type == 'heavy':
        train_dataset = HeavyChainDataset(
            data_path=data_path,
            dataconfig=data_config,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            batch_read_file=True
        )
    elif chain_type == 'light':
        train_dataset = LightChainDataset(data_path=data_path
                                          , lambda_dataconfig=data_config[1],
                                          kappa_dataconfig=data_config[0],
                                          batch_size=batch_size,
                                          max_sequence_length=max_sequence_length,
                                          batch_read_file=True)
    return train_dataset


def main():
    setup_logging()
    args = parse_args()

    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    logging.info("Configuration setup is done.")
    logging.info(f"The Script will train a {args.chain_type} AlignAIR")


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
    train_dataset = get_train_dataset(args.chain_type,args.train_dataset,args.batch_size,args.max_sequence_length)

    logging.info("Dataset is loaded and prepared.")

    # Initialize the model and trainer
    trainer = Trainer(
        model=HeavyChainAlignAIRR,
        dataset=train_dataset,
        epochs=args.epochs,
        steps_per_epoch=int(args.steps_per_epoch),
        verbose=1,
        classification_metric=[tf.keras.metrics.AUC(), tf.keras.metrics.AUC(), tf.keras.metrics.AUC()],
        regression_metric=tf.keras.losses.binary_crossentropy,
        log_to_file=True,
        log_file_name=args.model_name,
        log_file_path=logs_path,
        callbacks=[reduce_lr, model_checkpoint_callback],
        optimizers_params={"clipnorm": 1},
    )

    logging.info("Starting model training.")
    trainer.train()

    # Save model and weights
    path_to_model_weights = os.path.join(models_path, args.model_name)
    os.makedirs(path_to_model_weights, exist_ok=True)

    trainer.model.save_weights(os.path.join(path_to_model_weights, f'{args.model_name}_weights_final_epoch'))
    logging.info("Model training is complete and model is saved.")


if __name__ == "__main__":
    main()
