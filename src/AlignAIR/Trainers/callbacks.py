import tensorflow as tf


class AUCEpochEndCallback(tf.keras.callbacks.Callback):
    """
    Compute ROC-AUC on a validation dataset at the end of each epoch.

    - Iterates over a small, bounded number of validation batches
    - Supports multi-label heads (v_allele, j_allele, optional d_allele)
    - Logs results into the epoch `logs` dict so History/CSVLogger capture them

    Notes
    -----
    This avoids per-batch AUC updates during training, which can be costly and noisy
    for multi-label outputs, by computing a micro-averaged AUC from concatenated
    predictions/labels once per epoch.
    """

    def __init__(self, val_dataset: tf.data.Dataset, max_batches: int | None = None):
        super().__init__()
        self._val_dataset = val_dataset
        self._max_batches = max_batches

    @staticmethod
    def _compute_micro_auc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute micro-averaged AUC by flattening batch and class dims."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        auc = tf.keras.metrics.AUC(curve='ROC', name='auc')
        auc.update_state(y_true_f, y_pred_f)
        return auc.result()

    def on_epoch_end(self, epoch, logs=None):  # noqa: D401
        logs = logs or {}

        # Accumulators for heads we discover in the first batch
        heads = None
        y_true_acc = {}
        y_pred_acc = {}

        batches = self._val_dataset
        if self._max_batches is not None:
            batches = batches.take(self._max_batches)

        for x_batch, y_batch in batches:
            # Run a forward pass (no dropout)
            y_hat = self.model(x_batch, training=False)

            # Determine heads on first observed batch
            if heads is None:
                candidate_heads = ['v_allele', 'j_allele', 'd_allele']
                heads = [h for h in candidate_heads if (isinstance(y_batch, dict) and h in y_batch and h in y_hat)]
                # If nothing found, skip work for this epoch
                if not heads:
                    return
                for h in heads:
                    y_true_acc[h] = []
                    y_pred_acc[h] = []

            for h in heads:
                y_true_acc[h].append(tf.convert_to_tensor(y_batch[h]))
                y_pred_acc[h].append(tf.convert_to_tensor(y_hat[h]))

        # Compute AUCs and log
        name_map = {'v_allele': 'val_auc_v', 'j_allele': 'val_auc_j', 'd_allele': 'val_auc_d'}
        for h in (heads or []):
            if not y_true_acc[h]:
                continue
            y_true_cat = tf.concat(y_true_acc[h], axis=0)
            y_pred_cat = tf.concat(y_pred_acc[h], axis=0)
            try:
                auc_val = self._compute_micro_auc(y_true_cat, y_pred_cat)
                # Insert into logs for History/CSVLogger
                logs[name_map[h]] = float(auc_val.numpy())
            except Exception:  # pragma: no cover
                # Be resilient: don't fail the epoch if AUC computation breaks
                pass
