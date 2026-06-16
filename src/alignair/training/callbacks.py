"""Composable training callbacks."""
import csv


class Callback:
    def on_epoch_end(self, epoch: int, logs: dict) -> None: ...
    def on_train_end(self) -> None: ...


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 5, mode: str = "min"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = None
        self.wait = 0
        self.should_stop = False

    def _improved(self, current) -> bool:
        if self.best is None:
            return True
        return current < self.best if self.mode == "min" else current > self.best

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        if self.monitor not in logs:
            return
        current = logs[self.monitor]
        if self._improved(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True


class CSVLogger(Callback):
    def __init__(self, path: str):
        self.path = path
        self.rows: list[dict] = []

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        row = {"epoch": epoch}
        row.update({k: v for k, v in logs.items()})
        self.rows.append(row)
        self._flush()

    def _flush(self) -> None:
        fields = sorted({k for r in self.rows for k in r})
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

    def on_train_end(self) -> None:
        self._flush()


class CallbackList(Callback):
    def __init__(self, callbacks):
        self.callbacks = list(callbacks)

    @property
    def should_stop(self) -> bool:
        return any(getattr(c, "should_stop", False) for c in self.callbacks)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        for c in self.callbacks:
            c.on_epoch_end(epoch, logs)

    def on_train_end(self) -> None:
        for c in self.callbacks:
            c.on_train_end()
