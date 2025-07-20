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
# Datasets
from AlignAIR.Data import SingleChainDataset, MultiChainDataset
# Models
from AlignAIR.Models import SingleChainAlignAIR, MultiChainAlignAIR
# The new refactored trainer
from AlignAIR.Trainers import Trainer
# Data Configurations
import GenAIRR.data as genairr_data
from GenAIRR.dataconfig import DataConfig


def setup_logging():
    """Configures the root logger for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
        ]
    )


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A unified trainer for Single-Chain and Multi-Chain AlignAIRR models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Dataset Arguments ---
    parser.add_argument(
        "--train_datasets",
        required=True,
        nargs='+',
        help="One or more paths to training dataset files (e.g., data1.csv data2.csv)."
    )
    parser.add_argument(
        "--genairr_dataconfigs",
        required=True,
        nargs='+',
        help="One or more names of built-in GenAIRR dataconfigs or paths to .pkl files. Must match the order of --train_datasets."
    )
    # --- Training Arguments ---
    parser.add_argument("--session_path", required=True,
                        help="Base directory for saving all session output (models, logs, plots).")
    parser.add_argument("--model_name", default="AlignAIRR_Model", help="Name of the model for saving files.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--samples_per_epoch", type=int, default=1024,
                        help="Total number of samples to process per epoch.")
    # --- Model Arguments ---
    parser.add_argument("--max_sequence_length", type=int, default=576,
                        help="Maximum input sequence length for the model.")

    args = parser.parse_args()

    if len(args.train_datasets) != len(args.genairr_dataconfigs):
        parser.error("--train_datasets and --genairr_dataconfigs must have the same number of arguments.")

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
        raise ValueError(
            f"Invalid dataconfig: '{dataconfig_name_or_path}'. It is not a valid file path or a built-in GenAIRR dataconfig name.")


def build_losses_and_metrics(dataset) -> (dict, dict):
    """Dynamically builds the loss and metrics dictionaries based on the dataset type."""
    losses = {}
    metrics = {}

    # Determine the chain types and prefixes based on the dataset instance
    is_multi_chain = isinstance(dataset, MultiChainDataset)
    chain_types = dataset.chain_types if is_multi_chain else [dataset.dataconfig.metadata.chain_type]

    for chain_type in chain_types:
        prefix = f"{chain_type.value}_" if is_multi_chain else ""

        # --- Define metrics for each output head ---
        # Segmentation heads (MSE loss, MAE metric)
        for head in ['v_start', 'v_end', 'j_start', 'j_end']:
            losses[f'{prefix}{head}'] = 'mean_squared_error'
            metrics[f'{prefix}{head}'] = 'mean_absolute_error'

        # Allele classification heads (BCE loss, AUC and Accuracy metrics)
        for head in ['v_allele', 'j_allele']:
            losses[f'{prefix}{head}'] = 'binary_crossentropy'
            metrics[f'{prefix}{head}'] = [
                tf.keras.metrics.AUC(name='auc'),
                'binary_accuracy'
            ]

        # Add D-gene heads if they exist for this chain
        dataconfig = dataset.chain_datasets[chain_type] if is_multi_chain else dataset.dataconfig
        if dataconfig.metadata.has_d:
            for head in ['d_start', 'd_end']:
                losses[f'{prefix}{head}'] = 'mean_squared_error'
                metrics[f'{prefix}{head}'] = 'mean_absolute_error'

            losses[f'{prefix}d_allele'] = 'binary_crossentropy'
            metrics[f'{prefix}d_allele'] = [
                tf.keras.metrics.AUC(name=f'{prefix}d_allele_auc'),
                'binary_accuracy'
            ]

    # For MultiChain, there is an additional 'type' output
    if is_multi_chain:
        # The model has a single output for chain type classification
        losses['chain_type'] = 'categorical_crossentropy'
        metrics['chain_type'] = [
            tf.keras.metrics.CategoricalAUC(name='chain_type_auc'),
            'categorical_accuracy'
        ]

    return losses, metrics


def main():
    """Main function to run the training process."""
    setup_logging()
    args = parse_args()

    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # --- Determine Run Type (Single vs. Multi-chain) ---
    is_multi_chain = len(args.train_datasets) > 1
    run_type = "Multi-Chain" if is_multi_chain else "Single-Chain"
    logging.info(f"Starting {run_type} training run.")

    # --- Load DataConfigs ---
    dataconfigs = [get_dataconfig(dc) for dc in args.genairr_dataconfigs]

    # --- Create Dataset and Model ---
    if is_multi_chain:
        dataset = MultiChainDataset(
            data_paths=args.train_datasets,
            dataconfigs=dataconfigs,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            use_streaming=True
        )
        model_params = dataset.generate_model_params()
        model = MultiChainAlignAIR(**model_params)
    else:  # Single-chain
        dataset = SingleChainDataset(
            data_path=args.train_datasets[0],
            dataconfig=dataconfigs[0],
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            use_streaming=True
        )
        model_params = dataset.generate_model_params()
        model = SingleChainAlignAIR(**model_params)

    logging.info(f"Dataset and {model.__class__.__name__} created successfully.")

    # --- Compile Model ---
    # This is now done outside the Trainer, giving us full control.
    # The losses and metrics are generated dynamically to match the model's outputs.
    # Note: The model's custom train_step uses its own internal loss calculation.
    # The 'loss' parameter here is used by Keras to validate the structure, but the
    # actual loss values come from the model's `hierarchical_loss` method.
    # The 'metrics' are what Keras will track and display.
    _, metrics_dict = build_losses_and_metrics(dataset)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
        loss=None,  # Loss is handled inside the model's train_step
        metrics=metrics_dict
    )
    logging.info("Model compiled successfully.")

    # --- Define Callbacks ---
    # Note: Monitor a metric that is guaranteed to exist, like 'loss' or a specific metric.
    # For multi-chain, you might monitor 'val_IGH_v_allele_auc'.
    monitor_metric = 'v_allele_auc'

    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.9,
        patience=10,
        min_delta=0.001,
        verbose=1
    )

    checkpoint_path = Path(args.session_path) / f"{args.model_name}_best.weights.h5"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor=monitor_metric,
        mode="min"  # for loss
    )
    logging.info(f"Callbacks defined. Checkpointing best model based on '{monitor_metric}' to {checkpoint_path}")

    # --- Initialize Trainer and Start Training ---
    trainer = Trainer(model=model, session_path=args.session_path, model_name=args.model_name)

    trainer.train(
        train_dataset=dataset,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        callbacks=[reduce_lr, model_checkpoint_callback]
    )

    # --- Save Final Artifacts ---
    final_weights_path = Path(args.session_path) / f"{args.model_name}_final.weights.h5"
    model.save_weights(final_weights_path)
    trainer.save_training_history()
    trainer.plot_training_history()  # Plot all available metrics

    logging.info(f"Training complete. Final weights saved to {final_weights_path}")


if __name__ == "__main__":
    main()
