import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import uuid
import os
import torch.optim.lr_scheduler as lr_scheduler

class AlignAIRTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        loss_function,
        optimizer=torch.optim.Adam,
        optimizer_params=None,
        epochs=10,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_to_file=False,
        log_file_name=None,
        log_file_path=None,
        verbose=0,
        callbacks=None,
        batches_per_epoch=None,  # New parameter
    ):
        """
        PyTorch-based Trainer for training models with custom loss functions.

        Args:
            model (nn.Module): PyTorch model to be trained.
            dataset (Dataset): Dataset object for training.
            loss_function (nn.Module): Custom loss function.
            optimizer (Optimizer): Optimizer class, default is Adam.
            optimizer_params (dict): Parameters for the optimizer.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for DataLoader.
            device (str): Device to train on, default is CUDA if available.
            log_to_file (bool): Whether to log training history to a file.
            log_file_name (str): Name of the log file.
            log_file_path (str): Path to save the log file.
            verbose (int): Verbosity level.
            callbacks (list): List of custom callback functions.
            batches_per_epoch (int): Number of batches to process per epoch.
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.loss_function = loss_function
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.batches_per_epoch = batches_per_epoch  # Store the batches_per_epoch

        self.optimizer_params = optimizer_params or {"lr": 1e-3}
        self.optimizer = optimizer(self.model.parameters(), **self.optimizer_params)

        self.history = {"loss": []}
        self.log_to_file = log_to_file
        self.log_file_name = log_file_name or f"training_log_{uuid.uuid4().hex}.txt"
        self.log_file_path = log_file_path
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Learning Rate Scheduler
        self.scheduler_params = {"factor": 0.8, "patience": 3, "verbose": True}
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.scheduler_params)

    def train(self,wandb=None):
        """
        Train the model with the specified parameters.
        """
        self.model.train()
        loss_function = self.loss_function(self.model)
        total_steps = 0
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.data_loader):
                # Limit batches per epoch if specified
                if self.batches_per_epoch and batch_idx >= self.batches_per_epoch:
                    break

                # Move data to device
                inputs, targets = batch['x'], batch['y']
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device).unsqueeze(1) if v.dim() == 1 else v.to(self.device) for k, v in
                           targets.items()}

                # Forward pass
                predictions = self.model(inputs)
                loss, loss_components = loss_function(targets, predictions)
                loss_components = {k: v.item() for k, v in loss_components.items()}
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                total_steps+=1

                if wandb:
                    for component_name, component_value in loss_components.items():
                        wandb.log({f"Loss/{component_name}": component_value},
                                  step=total_steps)

                if self.verbose > 0 and batch_idx % self.verbose == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Batch [{batch_idx}], Loss: {loss.item()}")
                    for component in loss_components:
                        print(f"{component}: {loss_components[component]:.4f}")

            # Average loss for the epoch
            effective_batches = self.batches_per_epoch or len(self.data_loader)
            epoch_loss /= effective_batches
            self.history["loss"].append(epoch_loss)
            print(f"Epoch [{epoch+1}/{self.epochs}] completed with Avg Loss: {epoch_loss:.4f}")

            # Step scheduler with current loss
            self.scheduler.step(epoch_loss)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"Learning Rate after epoch {epoch + 1}: {current_lr:.6f}")
            print('-' * 50)

            # Run callbacks
            for callback in self.callbacks:
                callback(epoch, self.model, epoch_loss)

            # Log to file if enabled
            if self.log_to_file and self.log_file_path:
                os.makedirs(self.log_file_path, exist_ok=True)
                with open(os.path.join(self.log_file_path, self.log_file_name), "a") as log_file:
                    log_file.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}\n")

    def save_model(self, save_path):
        """
        Save the model weights.

        Args:
            save_path (str): Path to save the model weights.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, weights_path):
        """
        Load model weights.

        Args:
            weights_path (str): Path to the model weights.
        """
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        print(f"Model loaded from {weights_path}")

    def plot_history(self, save_path=None):
        """
        Plot the training history.

        Args:
            save_path (str, optional): Path to save the plot. If None, the plot will be displayed.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.history["loss"]) + 1), self.history["loss"], label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.grid()
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()