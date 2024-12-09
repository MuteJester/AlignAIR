import pickle
from uuid import uuid4
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from pathlib import Path

class Trainer:
    """
    A flexible and user-friendly trainer class for training TensorFlow models.

    Attributes:
        model (tf.keras.Model): The model to train.
        epochs (int): Number of epochs to train.
        steps_per_epoch (int): Steps per epoch.
        batch_size (int): Batch size for training.
        classification_metric (str or list): Metric(s) for classification tasks.
        regression_metric (str): Metric for regression tasks.
        callbacks (list): List of custom callbacks.
        pretrained_path (str): Path to pretrained model weights.
        log_to_file (bool): Whether to log training details to a file.
        log_file_name (str): Name of the log file.
        log_file_path (str): Path to save the log file.
    """
    def __init__(
        self,
        model,
        epochs,
        steps_per_epoch,
        batch_size,
        classification_metric='categorical_accuracy',
        regression_metric='mae',
        numbers_of_parallel_calls=8,
        pretrained_path=None,
        log_to_file=False,
        log_file_name=None,
        log_file_path=None,
        callbacks=None,
        optimizers=tf.keras.optimizers.Adam,
        optimizer_params=None,
        verbose=0,
    ):
        self.model = model
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.numbers_of_parallel_calls = numbers_of_parallel_calls
        self.classification_metric = classification_metric
        self.regression_metric = regression_metric
        self.callbacks = callbacks or []
        self.pretrained_path = pretrained_path
        self.log_to_file = log_to_file
        self.log_file_name = log_file_name
        self.log_file_path = log_file_path
        self.optimizers = optimizers
        self.optimizer_params = optimizer_params or {}
        self.verbose = verbose
        self.history = None

        if self.pretrained_path:
            self.load_model(self.pretrained_path)

    def compile_model(self, loss=None):
        """
        Compiles the model with the provided optimizer and metrics.
        """
        optimizer = self.optimizers(**self.optimizer_params)
        metrics = {
            'classification': self.classification_metric,
            'regression': self.regression_metric
        }
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_dataset):
        """
        Trains the model on the provided dataset.

        Args:
            train_dataset (DatasetBase): A custom dataset object that provides a TensorFlow dataset.
            validation_dataset (DatasetBase, optional): An optional validation dataset object.
        """
        # Preprocess the training dataset
        train_dataset = train_dataset.get_train_dataset().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#        train_dataset = train_dataset.map(lambda *args: args, num_parallel_calls=self.numbers_of_parallel_calls)

        def preprocess_data(*args):
            return args

        train_dataset = train_dataset.map(
            preprocess_data,
            num_parallel_calls=self.numbers_of_parallel_calls
        )

        # Add CSVLogger callback if logging is enabled
        if self.log_to_file:
            if not self.log_file_path:
                raise ValueError("Log file path must be specified when logging is enabled.")
            log_file_path = Path(self.log_file_path) / (self.log_file_name or f"{uuid4()}.csv")
            csv_logger = CSVLogger(log_file_path.as_posix(), append=True)
            self.callbacks.append(csv_logger)

        # Train the model
        self.history = self.model.fit(
            train_dataset,
            #validation_data=validation_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )

    def load_model(self, weights_path, max_seq_length=None):
        """
        Loads pretrained model weights. Builds the model if it is not already built.

        Args:
            weights_path (str): Path to the weights file.
            max_seq_length (tuple, optional): Input shape required to build the model if not built.
        """
        # Check if the model is built
        if not self.model.built:
            if max_seq_length is None:
                raise ValueError(
                    "Model is not built and input_shape is required to build the model."
                )
            # Build the model with the provided input shape
            self.model.build(input_shape={'tokenized_sequence': (max_seq_length, 1)})
            print(f"Model built with input shape: ({max_seq_length},1)")

        # Load weights
        self.model.load_weights(weights_path)
        print(f"Model weights loaded from {weights_path}.")

    def save_model(self, save_path):
        """
        Saves the model weights.
        """
        save_path = Path(save_path) / f"{uuid4()}_weights"
        self.model.save_weights(save_path.as_posix())
        print(f"Model saved at {save_path}")
        return save_path.as_posix()

    def save_history(self, save_path):
        """
        Saves the training history to a file.
        """
        save_path = Path(save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        print(f"Training history saved at {save_path}")

    def plot_history(self, save_path=None):
        """
        Plots the training history.
        """
        if not self.history:
            raise ValueError("No training history available to plot.")

        num_plots = len(self.history.history)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5 * num_rows))
        axes = axes.flatten()

        for i, (metric, values) in enumerate(self.history.history.items()):
            axes[i].plot(values)
            axes[i].set_title(metric)
            axes[i].set_xlabel("Epochs")
            axes[i].grid(True)

        for i in range(len(self.history.history), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Training plot saved to {save_path}")
        else:
            plt.show()
