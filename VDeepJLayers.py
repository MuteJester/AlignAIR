import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Attention, Conv2D, MaxPool2D, LeakyReLU
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    concatenate,
    Conv1D,
    MaxPool1D,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.layers import Multiply, Layer


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_batch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if "val" not in x]

        f, axs = plt.subplots(len(metrics), 1, figsize=(6, 15))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), self.metrics[metric], label=metric)
            #             if logs['val_' + metric]:
            #                 axs[i].plot(range(1, epoch + 2),
            #                             self.metrics['val_' + metric],
            #                             label='val_' + metric)

            #             axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


class CutoutLayer(Layer):
    def __init__(self, max_size, gene, **kwargs):
        super(CutoutLayer, self).__init__(**kwargs)
        self.max_size = max_size
        self.gene = gene

    def round_output(self, dense_output):
        max_value = tf.reduce_max(dense_output, axis=-1, keepdims=True)
        max_value = tf.clip_by_value(max_value, 0, self.max_size)
        max_value = tf.cast(max_value, dtype=tf.float32)
        return max_value

    def _call_v(self, inputs, batch_size):
        dense_start, dense_end = inputs
        x = self.round_output(dense_start)
        y = self.round_output(dense_end)
        indices = tf.keras.backend.arange(0, self.max_size, dtype=tf.float32)
        R = K.greater(indices, x) & K.less(indices, y)
        R = tf.cast(R, tf.float32)
        R = tf.reshape(R, shape=(batch_size, self.max_size, 1))
        return R

    def _call_d(self, inputs, batch_size):
        dense_start, dense_end = inputs
        x = self.round_output(dense_start)
        y = self.round_output(dense_end)
        indices = tf.keras.backend.arange(0, self.max_size, dtype=tf.float32)
        R = K.greater(indices, x) & K.less(indices, y)
        R = tf.cast(R, tf.float32)
        R = tf.reshape(R, shape=(batch_size, self.max_size, 1))
        return R

    def _call_j(self, inputs, batch_size):
        dense_start, dense_end = inputs
        x = self.round_output(dense_start)
        y = self.round_output(dense_end)
        indices = tf.keras.backend.arange(0, self.max_size, dtype=tf.float32)
        R = K.greater(indices, x) & K.less(indices, y)
        R = tf.cast(R, tf.float32)
        R = tf.reshape(R, shape=(batch_size, self.max_size, 1))
        return R

    def call(self, inputs):
        if self.gene == "V":
            batch_size = tf.shape(inputs[0])[0]
            return self._call_v(inputs, batch_size)
        elif self.gene == "D":
            batch_size = tf.shape(inputs[0])[0]
            return self._call_d(inputs, batch_size)
        elif self.gene == "J":
            batch_size = tf.shape(inputs[0])[0]
            return self._call_j(inputs, batch_size)

    def compute_output_shape(self, input_shape):
        return (None, self.max_size, 1)


class ExtractGeneMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractGeneMask, self).__init__(**kwargs)
        self.mul = Multiply()  # ([input_t,mid_cu])

    def call(self, inputs):
        tokenized_input, mask = inputs

        masked_input = self.mul([tokenized_input, mask])

        return masked_input


class ExtractedMaskEncode(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractedMaskEncode, self).__init__(**kwargs)
        self.conv_1 = Conv1D(
            16, 3, padding="same", activation="relu", kernel_initializer="he_uniform"
        )
        self.maxpool_1 = MaxPool1D(2)
        self.conv_2 = Conv1D(
            32, 3, padding="same", activation="relu", kernel_initializer="he_uniform"
        )
        self.attention_1 = Attention()
        self.conv_3 = Conv1D(
            8, 8, padding="same", activation="relu", kernel_initializer="he_uniform"
        )
        self.maxpool_2 = MaxPool1D(2)
        self.conv_4 = Conv1D(
            32, 3, padding="same", activation="relu", kernel_initializer="he_uniform"
        )
        self.attention_2 = Attention()
        self.conv_5 = Conv1D(
            8, 8, padding="same", activation="relu", kernel_initializer="he_uniform"
        )
        self.maxpool_3 = MaxPool1D(2)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.attention_1([x, x])
        x = self.maxpool_2(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.attention_2([x, x])
        x = self.maxpool_3(x)
        return x


class EncodingConcatBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncodingConcatBlock, self).__init__(**kwargs)
        self.concated = concatenate
        self.maxpool_1 = MaxPool1D(3)
        self.attention = Attention()
        self.flatten = Flatten()
        self.dense_1 = Dense(512, activation="relu")
        self.dropout_1 = Dropout(0.5)

    def call(self, inputs):
        x = self.concated(inputs)
        x = self.maxpool_1(x)
        x = self.attention([x, x])
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        return x


class Conv1D_and_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, max_pool):
        super(Conv1D_and_BatchNorm, self).__init__()
        self.conv_1d = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel,
            padding="same",
            kernel_initializer="he_uniform",
        )  # (inputs)
        self.batch_norm = BatchNormalization()  # (concatenated_path)
        self.activation = LeakyReLU()  # (concatenated_path)
        self.max_pool = MaxPool1D(max_pool)  # (concatenated_path)

    def call(self, inputs):
        x = self.conv_1d(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


def mod3_mse_loss(y_true, y_pred):
    y_true_casted = K.cast(y_true, dtype="float32")
    # Compute the residual of the predicted values modulo 3

    div_pred = K.cast(y_pred // 3.0, dtype="float32")
    mod_pred = y_pred - div_pred * 3

    div_true = K.cast(y_true_casted // 3.0, dtype="float32")
    mod_true = y_true_casted - div_true * 3

    residual = K.abs(mod_pred - mod_true)

    # Penalize the deviation from the nearest multiple of 3
    mod3_loss = K.mean(K.switch(residual <= 1, residual, 3 - residual))

    # Compute the MSE loss
    mse_loss = K.mean(K.square(y_pred - y_true_casted))

    # Combine the two losses using a weighted sum
    mod3_weight, mse_weight = 0.3, 0.7  # adjust the weights as needed
    total_loss = mod3_weight * mod3_loss + mse_weight * mse_loss

    return total_loss


def mod3_mse_regularization(y_true, y_pred):
    """
    Computes the mean squared error loss with regularization to enforce predictions to be modulo 3.
    """
    mse = K.mean(K.square(y_true - y_pred))
    mod_pred = K.abs(y_pred - 3 * K.cast(K.round(y_pred / 3), "float32"))
    mod_loss = K.mean(K.square(mod_pred))
    return mse + mod_loss
