"""PyTorch trainer for AlignAIR models."""
import logging

import torch

from .config import TrainingConfig, seed_everything
from .callbacks import CallbackList

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, loss_fn, config: TrainingConfig, device: str | None = None,
                 optimizer=None):
        self.config = config
        seed_everything(config.seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = optimizer or torch.optim.Adam(
            params, lr=config.lr, weight_decay=config.weight_decay)
        self.use_amp = config.use_amp and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _to_device(self, d):
        return {k: v.to(self.device) for k, v in d.items()}

    def train_step(self, x, y) -> dict:
        self.model.train()
        x, y = self._to_device(x), self._to_device(y)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device, enabled=self.use_amp):
            y_pred = self.model(x["tokenized_sequence"]).as_dict()
            total, components = self.loss_fn(y, y_pred)
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            self.config.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.apply_constraints()
        self.loss_fn.apply_constraints()
        logs = {"loss": float(total.detach().cpu())}
        logs.update({k: float(v.cpu()) for k, v in components.items()})
        return logs

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        totals, n = {}, 0
        for x, y in loader:
            x, y = self._to_device(x), self._to_device(y)
            y_pred = self.model(x["tokenized_sequence"]).as_dict()
            total, components = self.loss_fn(y, y_pred)
            n += 1
            totals["val_loss"] = totals.get("val_loss", 0.0) + float(total.cpu())
        if n:
            totals = {k: v / n for k, v in totals.items()}
        return totals

    def train_epoch(self, loader) -> dict:
        running, steps = 0.0, 0
        for i, (x, y) in enumerate(loader):
            logs = self.train_step(x, y)
            running += logs["loss"]
            steps += 1
            if self.config.steps_per_epoch and steps >= self.config.steps_per_epoch:
                break
        return {"loss": running / max(steps, 1)}

    def fit(self, train_loader, val_loader=None, callbacks=None) -> list[dict]:
        cb = CallbackList(callbacks or [])
        history = []
        for epoch in range(self.config.epochs):
            train_logs = self.train_epoch(train_loader)
            logs = dict(train_logs)
            if val_loader is not None:
                logs.update(self.evaluate(val_loader))
            history.append(logs)
            cb.on_epoch_end(epoch, logs)
            if cb.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        cb.on_train_end()
        return history

    def save_checkpoint(self, path: str, epoch: int) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "config": self.config.to_dict(),
            "torch_rng": torch.get_rng_state(),
        }, path)

    def load_checkpoint(self, path: str) -> dict:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.loss_fn.load_state_dict(state["loss_fn"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        if "torch_rng" in state:
            # map_location moves every checkpoint tensor to self.device; the RNG state
            # must be a CPU ByteTensor, so bring it back before restoring.
            torch.set_rng_state(state["torch_rng"].to("cpu", torch.uint8))
        return {"epoch": state["epoch"], "config": state["config"]}
