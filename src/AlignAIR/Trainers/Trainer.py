import pickle
from uuid import uuid4
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import CSVLogger
from pathlib import Path
import logging
import time
from typing import Optional, List, Any, cast

from AlignAIR.Serialization.model_bundle import TrainingMeta  # Step 6 integration

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

    def __init__(self, model: keras.Model, session_path: str, model_name: str = "model"):
        """
        Initializes the Trainer.

        Args:
            model (tf.keras.Model): The Keras model to be trained. It is expected
                                    to be compiled before being passed to the trainer.
            session_path (str): The base directory where all training artifacts
                                (logs, history, plots) will be saved.
            model_name (str): A name for the model, used for naming saved files.
        """
        if not isinstance(model, keras.Model):
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
              callbacks: Optional[list] = None,
              save_pretrained: bool = False,
              bundle_dir: Optional[str] = None,
              training_notes: Optional[str] = None,
              export_saved_model: bool = False,
              include_logits_in_saved_model: bool = False):
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
        # Datasets already apply prefetch internally; avoid double prefetch to reduce buffer memory.
        tf_train_dataset = train_dataset.get_train_dataset()

        tf_validation_dataset = None
        validation_steps = None
        if validation_dataset:
            tf_validation_dataset = validation_dataset.get_train_dataset()
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
        start_time = time.time()
        self.history = self.model.fit(
            tf_train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=tf_validation_dataset,
            validation_steps=validation_steps,
            callbacks=all_callbacks,
            verbose='auto'
        )
        wall_time = int(time.time() - start_time)
        logger.info("Training finished.")

        # --- Optional bundle save ---
        if save_pretrained:
            try:
                # History can contain numpy/tf dtypes; coerce everything to plain Python types
                hist = self.history.history if self.history else {}
                raw_losses = hist.get('loss', [])
                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        try:
                            # Some tensors have .numpy(); fall back to that
                            return float(getattr(x, 'numpy')())
                        except Exception:
                            return None
                losses = [lv for lv in (_to_float(x) for x in raw_losses) if lv is not None]
                best_loss = min(losses) if losses else None
                best_epoch = int(losses.index(cast(float, best_loss))) if losses else None
                final_loss = losses[-1] if losses else None
                metrics_summary = {}
                for k, v in hist.items():
                    if not v or k.startswith('val_'):
                        continue
                    val = _to_float(v[-1])
                    if val is not None:
                        metrics_summary[k] = val
                # learning rate (handle schedules)
                try:
                    lr = self.model.optimizer.learning_rate
                    if hasattr(lr, 'numpy'):
                        lr_value = float(lr.numpy())
                    else:
                        lr_value = str(lr)
                except Exception:
                    lr_value = None
                # mixed precision detection
                try:
                    mp = getattr(self.model, 'dtype_policy', None)
                    mixed_precision = bool(mp and 'float16' in str(mp))
                except Exception:
                    mixed_precision = None
                meta = TrainingMeta(
                    epochs_trained=int(epochs),
                    final_epoch=int(epochs - 1),
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    final_loss=final_loss,
                    metrics_summary=metrics_summary,
                    wall_time_seconds=int(wall_time) if wall_time is not None else None,
                    batch_size=int(batch_size),
                    samples_per_epoch=int(samples_per_epoch),
                    optimizer_class=self.model.optimizer.__class__.__name__,
                    learning_rate=str(lr_value),
                    mixed_precision=mixed_precision,
                    extra={'notes': training_notes} if training_notes else {}
                )
                # derive bundle dir if not provided
                if bundle_dir is None:
                    bundle_dir = (self.session_path / f"{self.model_name}_bundle").as_posix()
                if hasattr(self.model, 'save_pretrained'):
                    # Pass through SavedModel export flags if model supports them
                    try:
                        # export_saved_model is now always True in the SavedModel-first flow;
                        # we keep the argument for backward compatibility but do not pass it through.
                        self.model.save_pretrained(
                            bundle_dir,
                            training_meta=meta,
                            include_logits_in_saved_model=include_logits_in_saved_model
                        )
                    except TypeError:
                        # Backward compatibility: method without new args
                        self.model.save_pretrained(bundle_dir, training_meta=meta)
                    logger.info("Saved pretrained bundle to %s", bundle_dir)
                else:
                    logger.warning("Model has no save_pretrained; skipping bundle save.")
            except Exception as e:
                logger.error("Failed to save pretrained bundle: %s", e, exc_info=True)
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

    def plot_training_history(self, metrics_to_plot: Optional[list] = None):
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
