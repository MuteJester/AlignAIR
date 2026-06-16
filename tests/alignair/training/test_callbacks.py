import csv
from alignair.training.callbacks import EarlyStopping, CSVLogger, CallbackList


def test_early_stopping_triggers():
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    es.on_epoch_end(0, {"val_loss": 1.0})
    assert not es.should_stop
    es.on_epoch_end(1, {"val_loss": 1.1})  # worse (1)
    es.on_epoch_end(2, {"val_loss": 1.2})  # worse (2) -> stop
    assert es.should_stop


def test_early_stopping_resets_on_improve():
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    es.on_epoch_end(0, {"val_loss": 1.0})
    es.on_epoch_end(1, {"val_loss": 1.1})  # worse (wait 1)
    es.on_epoch_end(2, {"val_loss": 0.5})  # improve -> reset wait to 0
    es.on_epoch_end(3, {"val_loss": 0.6})  # worse (wait 1) -> still < patience 2
    assert not es.should_stop


def test_csv_logger_writes(tmp_path):
    p = tmp_path / "log.csv"
    logger = CSVLogger(str(p))
    logger.on_epoch_end(0, {"loss": 1.0, "val_loss": 0.9})
    logger.on_epoch_end(1, {"loss": 0.8, "val_loss": 0.7})
    logger.on_train_end()
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[1]["loss"] == "0.8"


def test_callback_list_dispatches():
    es = EarlyStopping(monitor="val_loss", patience=0, mode="min")
    cl = CallbackList([es])
    cl.on_epoch_end(0, {"val_loss": 1.0})
    cl.on_epoch_end(1, {"val_loss": 1.1})
    assert cl.should_stop
