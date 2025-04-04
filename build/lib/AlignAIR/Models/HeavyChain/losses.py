import tensorflow as tf
import tensorflow.keras.backend as K


def d_loss(y_true, y_pred, penalty_factor=1.0, last_label_penalty_factor=1.0):
    # Binary crossentropy
    bce = tf.keras.metrics.binary_crossentropy(y_true, y_pred)

    # Calculate the total sum of the prediction vectors
    total_sum = K.sum(y_pred, axis=-1)

    # Calculate the threshold which is 90% of the total sum
    threshold = 0.9 * total_sum

    # Count how many labels are above this threshold
    labels_above_threshold = K.sum(K.cast(y_pred > threshold[:, None], tf.float32), axis=1)

    # Apply penalty if count of labels above threshold is greater than 5
    extra_penalty = penalty_factor * K.cast(labels_above_threshold > 5, tf.float32)

    # Additional penalty if the last label's likelihood is above 0.5 and any other label is above zero
    last_label_high_confidence = K.cast(K.greater(y_pred[:, -1], 0.5), tf.float32)  # Check if last label > 0.5
    other_labels_above_zero = K.cast(K.any(K.greater(y_pred[:, :-1], 0), axis=1),
                                     tf.float32)  # Check if any other label > 0
    last_label_penalty = last_label_penalty_factor * last_label_high_confidence * other_labels_above_zero

    # Combined loss with both penalties
    return bce + extra_penalty + last_label_penalty
