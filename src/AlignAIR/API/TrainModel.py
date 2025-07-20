import os
import argparse
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import logging
from pathlib import Path

# --- Import your project's modules ---
from AlignAIR.Data import SingleChainDataset, MultiChainDataset, MultiDataConfigContainer
from AlignAIR.Models import SingleChainAlignAIR, MultiChainAlignAIR
from AlignAIR.Trainers import Trainer
import GenAIRR.data as genairr_data
from GenAIRR.dataconfig import DataConfig


def setup_logging():
    """Configures the root logger for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A unified trainer for Single-Chain and Multi-Chain AlignAIRR models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train_datasets", required=True, nargs='+',
                        help="One or more paths to training dataset files.")
    parser.add_argument("--genairr_dataconfigs", required=True, nargs='+',
                        help="One or more GenAIRR dataconfigs. Must match the order of --train_datasets.")
    parser.add_argument("--validation_datasets", nargs='+',
                        help="[Optional] One or more paths to validation dataset files.")
    parser.add_argument("--validation_dataconfigs", nargs='+',
                        help="[Optional] GenAIRR dataconfigs for validation. Must match order of validation datasets.")
    parser.add_argument("--session_path", required=True, help="Base directory for saving all session output.")
    parser.add_argument("--model_name", default="AlignAIRR_Model", help="Name of the model for saving files.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--samples_per_epoch", type=int, default=1024,
                        help="Total number of samples to process per epoch.")
    parser.add_argument("--max_sequence_length", type=int, default=576,
                        help="Maximum input sequence length for the model.")

    args = parser.parse_args()
    # Argument validation
    if len(args.train_datasets) != len(args.genairr_dataconfigs):
        parser.error("--train_datasets and --genairr_dataconfigs must have the same number of arguments.")
    if bool(args.validation_datasets) != bool(args.validation_dataconfigs):
        parser.error("--validation_datasets and --validation_dataconfigs must be provided together.")
    if args.validation_datasets and (len(args.validation_datasets) != len(args.train_datasets)):
        parser.error("The number of training and validation datasets must match for a consistent model architecture.")
    return args


def get_dataconfig(dataconfig_name_or_path: str) -> DataConfig:
    """Loads a DataConfig from a built-in name or a .pkl file."""
    if Path(dataconfig_name_or_path).is_file():
        logging.info(f"Loading DataConfig from file: {dataconfig_name_or_path}")
        with open(dataconfig_name_or_path, 'rb') as f:
            return pickle.load(f)
    elif hasattr(genairr_data, dataconfig_name_or_path):
        logging.info(f"Loading built-in GenAIRR DataConfig: {dataconfig_name_or_path}")
        return getattr(genairr_data, dataconfig_name_or_path)
    else:
        raise ValueError(f"Invalid dataconfig: '{dataconfig_name_or_path}'.")


def build_metrics_dict(dataset) -> dict:
    """
    Dynamically builds the metrics dictionary.

    *** THIS IS THE CORRECTED LOGIC ***
    This version correctly handles the concatenated (non-prefixed) output
    of the MultiChainDataset.
    """
    metrics = {}
    is_multi_chain = isinstance(dataset, MultiChainDataset)

    # For both single and multi-chain, the output keys are the same (non-prefixed)
    # because the batches are concatenated.
    for head in ['v_start', 'v_end', 'j_start', 'j_end']:
        metrics[head] = 'mean_absolute_error'

    for head in ['v_allele', 'j_allele']:
        metrics[head] = [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy']

    # Check if any of the underlying dataconfigs has a D-gene
    has_d = dataset.has_d if hasattr(dataset, 'has_d') else dataset.dataconfig.metadata.has_d
    if has_d:
        for head in ['d_start', 'd_end']:
            metrics[head] = 'mean_absolute_error'
        metrics['d_allele'] = [tf.keras.metrics.AUC(name='auc'), 'binary_accuracy']

    # The MultiChain model has an additional 'chain_type' output
    if is_multi_chain:
        metrics['chain_type'] = ['categorical_accuracy', tf.keras.metrics.AUC(name='auc')]

    return metrics


def create_dataset(is_multi_chain, data_paths, dataconfigs, args):
    """Factory function to create the correct dataset object."""
    if is_multi_chain:
        return MultiChainDataset(
            data_paths=data_paths, dataconfigs=dataconfigs, batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length, use_streaming=True
        )
    else:
        return SingleChainDataset(
            data_path=data_paths[0], dataconfig=dataconfigs, batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length, use_streaming=True
        )


def main():
    """Main function to run the training process."""
    setup_logging()
    args = parse_args()

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    is_multi_chain = len(args.train_datasets) > 1
    run_type = "Multi-Chain" if is_multi_chain else "Single-Chain"
    logging.info(f"Starting {run_type} training run.")

    train_dataconfigs = [get_dataconfig(dc) for dc in args.genairr_dataconfigs]
    train_dataconfigs = MultiDataConfigContainer(train_dataconfigs)
    train_dataset = create_dataset(is_multi_chain, args.train_datasets, train_dataconfigs, args)

    validation_dataset = None
    if args.validation_datasets:
        logging.info("Creating validation dataset.")
        validation_dataconfigs = [get_dataconfig(dc) for dc in args.validation_dataconfigs]
        validation_dataconfigs = MultiDataConfigContainer(validation_dataconfigs)
        validation_dataset = create_dataset(is_multi_chain, args.validation_datasets, validation_dataconfigs, args)

    model_params = train_dataset.generate_model_params()
    model = MultiChainAlignAIR(**model_params) if is_multi_chain else SingleChainAlignAIR(**model_params)
    logging.info(f"Dataset and {model.__class__.__name__} created successfully.")

    metrics_dict = build_metrics_dict(train_dataset)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
        loss=None,  # Loss is handled inside the model's custom train_step
        metrics=metrics_dict
    )
    logging.info("Model compiled successfully.")

    monitor_metric = 'val_loss' if validation_dataset else 'loss'

    reduce_lr = ReduceLROnPlateau(monitor=monitor_metric, factor=0.9, patience=10, min_delta=0.001, verbose=1,
                                  mode='min')

    checkpoint_path = Path(args.session_path) / f"{args.model_name}_best.weights.h5"
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True,
                                                monitor=monitor_metric, mode="min")
    logging.info(f"Callbacks defined. Checkpointing best model based on '{monitor_metric}' to {checkpoint_path}")

    trainer = Trainer(model=model, session_path=args.session_path, model_name=args.model_name)

    trainer.train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        callbacks=[reduce_lr, model_checkpoint_callback]
    )

    final_weights_path = Path(args.session_path) / f"{args.model_name}_final.weights.h5"
    model.save_weights(final_weights_path)
    trainer.save_training_history()
    trainer.plot_training_history()

    logging.info(f"Training complete. Final weights saved to {final_weights_path}")


if __name__ == "__main__":
    main()
