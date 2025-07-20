import pickle
from uuid import uuid4
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from pathlib import Path
import logging

# It's good practice to have a logger instance for the module
logger = logging.getLogger(__name__)


class Trainer:
    """
    A professional, reusable trainer for TensorFlow Keras models.

    This trainer is designed with a clear separation of concerns:
    - The Model is responsible for its architecture, compilation (loss, metrics), and weights.
    - The Dataset is responsible for providing data.
    - The Trainer is responsible for orchestrating the training loop (`fit`) and managing artifacts (logs, history).

    This design makes the Trainer agnostic to the model's specific inputs or outputs,
    allowing it to seamlessly train both SingleChainAlignAIR and MultiChainAlignAIR models.
    """

    def __init__(self, model: tf.keras.Model, session_path: str, model_name: str = "model"):
        """
        Initializes the Trainer.

        Args:
            model (tf.keras.Model): The Keras model to be trained. It is expected
                                    to be compiled before being passed to the trainer.
            session_path (str): The base directory where all training artifacts
                                (logs, history, plots) will be saved.
            model_name (str): A name for the model, used for naming saved files.
        """
        if not isinstance(model, tf.keras.Model):
            raise TypeError("The 'model' argument must be an instance of tf.keras.Model.")
        if not model.optimizer:
            raise ValueError("The model must be compiled before being passed to the Trainer.")

        self.model = model
        self.session_path = Path(session_path)
        self.model_name = model_name
        self.history = None

        # Create the session directory if it doesn't exist
        self.session_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trainer initialized. Artifacts will be saved in: {self.session_path}")

    def train(self,
              train_dataset,
              epochs: int,
              samples_per_epoch: int,
              batch_size: int,
              validation_dataset=None,
              callbacks: list = None):
        """
        Trains the model on the provided dataset.

        Args:
            train_dataset: A dataset object with a `get_train_dataset()` method that
                           returns a `tf.data.Dataset`.
            epochs (int): The total number of epochs for training.
            samples_per_epoch (int): The total number of samples to process per epoch.
            batch_size (int): The size of each batch.
            validation_dataset (optional): A dataset object for validation.
            callbacks (list, optional): A list of Keras callbacks to use during training.
        """
        # --- Prepare Datasets ---
        tf_train_dataset = train_dataset.get_train_dataset().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        tf_validation_dataset = None
        validation_steps = None
        if validation_dataset:
            tf_validation_dataset = validation_dataset.get_train_dataset().prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            try:
                validation_steps = len(validation_dataset) // batch_size
                if validation_steps == 0:
                    validation_steps = 1  # Ensure at least one validation step if dataset is small
            except (TypeError, AttributeError):
                logger.warning(
                    "Validation dataset has no __len__ method. Validation will run on the full dataset once per epoch.")

        # --- Prepare Callbacks ---
        all_callbacks = list(callbacks) if callbacks else []

        # Add a CSVLogger by default for good practice
        log_file = self.session_path / f"{self.model_name}_training_log.csv"
        csv_logger = CSVLogger(log_file.as_posix(), append=True)
        all_callbacks.append(csv_logger)
        logger.info(f"Training logs will be saved to {log_file}")

        # --- Calculate Steps ---
        if samples_per_epoch < batch_size:
            raise ValueError(
                f"samples_per_epoch ({samples_per_epoch}) must be greater than or equal to batch_size ({batch_size}).")
        steps_per_epoch = samples_per_epoch // batch_size

        logger.info(f"Starting training for {epochs} epochs with {steps_per_epoch} steps per epoch.")

        # --- Run Training ---
        self.history = self.model.fit(
            tf_train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=tf_validation_dataset,
            validation_steps=validation_steps,
            callbacks=all_callbacks,
            verbose=1
        )

        logger.info("Training finished.")
        return self.history

    def save_training_history(self):
        """Saves the training history object to a pickle file."""
        if not self.history:
            logger.warning("No training history to save.")
            return

        history_path = self.session_path / f"{self.model_name}_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        logger.info(f"Training history saved to {history_path}")

    def plot_training_history(self, metrics_to_plot: list = None):
        """
        Plots the training history for specified metrics.

        Args:
            metrics_to_plot (list, optional): A list of metric names to plot. If None, plots all.
        """
        if not self.history:
            logger.warning("No training history available to plot.")
            return

        history_dict = self.history.history
        plot_keys = metrics_to_plot or list(history_dict.keys())

        # Automatically pair validation metrics with training metrics
        filtered_keys = []
        for key in plot_keys:
            if key in history_dict:
                filtered_keys.append(key)
            val_key = f"val_{key}"
            if val_key in history_dict:
                filtered_keys.append(val_key)

        history_to_plot = {k: history_dict[k] for k in sorted(list(set(filtered_keys)))}

        if not history_to_plot:
            logger.warning("None of the requested metrics were found in the training history.")
            return

        num_plots = len(history_to_plot)
        num_cols = min(3, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), squeeze=False)
        axes = axes.flatten()

        for i, (metric, values) in enumerate(history_to_plot.items()):
            axes[i].plot(values, label=metric)
            axes[i].set_title(metric.replace("_", " ").title())
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Value")
            axes[i].legend()
            axes[i].grid(True)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plot_path = self.session_path / f"{self.model_name}_training_plot.png"
        plt.savefig(plot_path)
        logger.info(f"Training plot saved to {plot_path}")
        plt.close()
