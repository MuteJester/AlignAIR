from typing import Any

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, LeakyReLU, MaxPool1D, Add, Flatten, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Attention, Conv2D, MaxPool2D, LeakyReLU
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import Flatten, concatenate, Conv1D, MaxPool1D, BatchNormalization
from tensorflow.keras.layers import Multiply, Layer, SeparableConv1D
import numpy as np
from tensorflow.keras.constraints import Constraint

import importlib
import os
from collections import defaultdict
from tensorflow.keras import layers, regularizers, constraints, metrics
from functools import partial


class SegmentationHead(layers.Layer):
    """Returns two tensors: (start, end) each with shape (B, 1)."""
    def __init__(self, name: str,
                 act: str  = "gelu",
                 init: str = "glorot_uniform"):
        super().__init__(name=name)
        dense = partial(
            layers.Dense,
            units=1,
            activation=act,
            kernel_initializer=init,
            kernel_constraint=constraints.unit_norm(),
        )
        self.start = dense(name=f"{name}_start")
        self.end   = dense(name=f"{name}_end")

    def call(
        self,
        inputs: tf.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return self.start(inputs), self.end(inputs)



class GeneClassificationHead(layers.Layer):
    """Latent dense → sigmoid output."""
    def __init__(
        self,
        alleles: int,
        cfg,
        name: str,
        reg = None,
    ):
        super().__init__(name=name)

        latent_size = cfg.latent_factor * alleles
        self.mid = layers.Dense(
            latent_size,
            activation=cfg.mid_activation,
            kernel_initializer=cfg.weight_init,
            name=f"{name}_mid",
            kernel_regularizer=reg,
        )
        self.out = layers.Dense(
            alleles,
            activation="sigmoid",
            name=name,
        )

    # full, type-correct override
    def call(self,inputs: tf.Tensor,*args: Any, **kwargs: Any,) -> tf.Tensor:
        x = self.mid(inputs)
        return self.out(x)



class ClipConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class Conv1D_and_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernel=3, max_pool=2, activation=None, initializer=None, **kwargs):
        super(Conv1D_and_BatchNorm, self).__init__(**kwargs)
        initializer_ = 'glorot_uniform' if initializer is None else initializer
        self.conv_2d = Conv1D(filters, kernel, padding='same',
                              kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_)
        self.conv_2d_2 = Conv1D(filters, kernel, padding='same',
                                kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_)

        self.conv_2d_3 = Conv1D(filters, kernel, padding='same',
                                kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_)

        self.batch_norm = BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02)

        if activation is None:
            self.activation = LeakyReLU()
        else:
            self.activation = activation
        self.max_pool = MaxPool1D(max_pool)

    def call(self, inputs):
        x = self.conv_2d(inputs)
        x = self.conv_2d_2(x)
        x = self.conv_2d_3(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class ConvResidualFeatureExtractionBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size=64, num_conv_batch_layers=5, kernel_size=5, max_pool_size=2,
                 conv_activation=None, out_shape=576, initializer=None, **kwargs):
        super(ConvResidualFeatureExtractionBlock, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.num_conv_batch_layers = num_conv_batch_layers
        self.kernel_size = kernel_size
        self.max_pool_size = max_pool_size
        self.conv_activation = conv_activation
        self.out_shape = out_shape

        self.initializer = (
            tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
            if initializer is None else initializer
        )

    def build(self, input_shape):
        # Dynamically initialize layers based on input shape
        if isinstance(self.kernel_size, int):
            self.conv_layers = [
                Conv1D_and_BatchNorm(
                    filters=self.filter_size,
                    kernel=self.kernel_size,
                    max_pool=self.max_pool_size,
                    initializer=self.initializer,
                    activation=self.conv_activation,
                ) for _ in range(self.num_conv_batch_layers)
            ]
            self.residual_channel = Conv1D(
                self.filter_size,
                self.kernel_size,
                padding='same',
                kernel_regularizer=regularizers.l2(0.01),
                kernel_initializer=self.initializer
            )
        elif isinstance(self.kernel_size, list):
            self.conv_layers = [
                Conv1D_and_BatchNorm(
                    filters=self.filter_size,
                    kernel=ks,
                    max_pool=self.max_pool_size,
                    initializer=self.initializer,
                    activation=self.conv_activation,
                ) for ks in self.kernel_size[:-1]
            ]
            self.residual_channel = Conv1D(
                self.filter_size,
                self.kernel_size[-1],
                padding='same',
                kernel_regularizer=regularizers.l2(0.01),
                kernel_initializer=self.initializer
            )

        self.max_pool_layers = [MaxPool1D(2) for _ in range(self.num_conv_batch_layers)]
        self.activation_layers = [LeakyReLU() for _ in range(self.num_conv_batch_layers)]
        self.add_layers = [Add() for _ in range(self.num_conv_batch_layers)]

        self.dense_reshaper = Dense(self.out_shape, activation='linear')
        self.segmentation_feature_flatten = Flatten()

        # Call parent build to set up tracking
        super(ConvResidualFeatureExtractionBlock, self).build(input_shape)

    def call(self, embeddings):
        # Residual stream
        residual_stream = self.residual_channel(embeddings)
        residual_end = self.max_pool_layers[0](residual_stream)

        feature_stream = self.conv_layers[0](embeddings)
        residual_end = self.add_layers[0]([feature_stream, residual_end])
        residual_end = self.activation_layers[0](residual_end)
        residual_end = self.max_pool_layers[0](residual_end)

        for index in range(1, len(self.max_pool_layers)):
            # Get F(x)
            feature_stream = self.conv_layers[index](residual_end)
            residual_end = self.max_pool_layers[index](residual_end)
            residual_end = self.add_layers[index]([feature_stream, residual_end])
            residual_end = self.activation_layers[index](residual_end)

        residual_end = self.segmentation_feature_flatten(residual_end)
        residual_end = self.dense_reshaper(residual_end)
        return residual_end






def global_genotype():
    try:
        path_to_data = importlib.resources.files(
            'airrship').joinpath("data")
    except AttributeError:
        with importlib.resources.path('airrship', 'data') as p:
            path_to_data = p
    v_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHV.fasta")
    d_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHD.fasta")
    j_alleles = create_allele_dict(
        f"{path_to_data}/imgt_human_IGHJ.fasta")

    vdj_allele_dicts = {"V": v_alleles,
                        "D": d_alleles,
                        "J": j_alleles}

    chromosome1, chromosome2 = defaultdict(list), defaultdict(list)
    for segment in ["V", "D", "J"]:
        allele_dict = vdj_allele_dicts[segment]
        for gene in allele_dict.values():
            for allele in gene:
                chromosome1[segment].append(allele)
                chromosome2[segment].append(allele)

    locus = [chromosome1, chromosome2]
    return locus


class RegularizedConstrainedLogVar(tf.keras.layers.Layer):
    def __init__(self,layer_name='log_var', initial_value=1.0, min_log_var=-3, max_log_var=1, regularizer_weight=0.01):
        super().__init__()
        self.initial_value = initial_value
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.regularizer_weight = regularizer_weight
        self.layer_name = layer_name

    def build(self, input_shape):
        # Dynamically initialize the log_var weight
        self.log_var = self.add_weight(
            name=self.layer_name,
            shape=(),  # Scalar weight
            initializer=tf.keras.initializers.Constant(value=tf.math.log(self.initial_value)),
            constraint=ClipConstraint(self.min_log_var, self.max_log_var),
            trainable=True
        )
        # Call the parent's build method
        super(RegularizedConstrainedLogVar, self).build(input_shape)

    def call(self, inputs):
        # Add a regularization loss
        regularization_loss = self.regularizer_weight * tf.nn.relu(-self.log_var - 2)  # Soft threshold at log(var)=2
        self.add_loss(regularization_loss)
        # Return precision as exp(-log(var))
        return tf.exp(-self.log_var)


def soft_mask(indices, start, end, K):
    # For positions greater than start but less than start + K, ramp up linearly
    ramp_up = (indices - start) / K
    ramp_up = tf.clip_by_value(ramp_up, 0, 1)

    # For positions less than end but greater than end - K, ramp down linearly
    ramp_down = (end - indices) / K
    ramp_down = tf.clip_by_value(ramp_down, 0, 1)

    # Use the hard mask as the base but modulate it with the soft ramping on each side
    mask = tf.minimum(ramp_up, ramp_down)

    # For positions between start + K and end - K, the mask should be 1
    mask = tf.where(
        (indices >= start + K) & (indices <= end - K),
        1.0,
        mask
    )
    return mask


def interval_iou(interval1, interval2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # interval1 and interval2 should be 1x2 tensors [start, end]

    # Get the coordinates of intervals
    b1_start, b1_end = interval1[0], interval1[1]
    b2_start, b2_end = interval2[0], interval2[1]

    # Intersection area
    inter = K.maximum(0.0, K.minimum(b1_end, b2_end) - K.maximum(b1_start, b2_start))

    # Union Area
    len1 = b1_end - b1_start + eps
    len2 = b2_end - b2_start + eps
    union = len1 + len2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        c_len = K.maximum(b1_end, b2_end) - K.minimum(b1_start, b2_start)  # convex (smallest enclosing interval) length
        if CIoU or DIoU:
            c2 = c_len ** 2 + eps  # convex length squared
            rho2 = ((b2_start + b2_end - b1_start - b1_end) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:
                v = (4 / (math.pi ** 2)) * K.square(tf.math.atan(len2 / (len1 + eps)))
                alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:
            c_area = c_len + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU



class CutoutLayer(tf.keras.layers.Layer):
    def __init__(self, max_size, gene, **kwargs):
        super(CutoutLayer, self).__init__(**kwargs)
        self.max_size = max_size
        self.gene = gene

    def build(self, input_shape):
        """
        Initialize any configuration dependent on input shape.
        """
        # Create a tensor for indices (common for all gene types)
        self.indices = tf.keras.backend.arange(0, self.max_size, dtype=tf.float32)

        # Call parent's build method to finalize setup
        super(CutoutLayer, self).build(input_shape)

    def round_output(self, dense_output):
        max_value = tf.reduce_max(dense_output, axis=-1, keepdims=True)
        max_value = tf.clip_by_value(max_value, 0, self.max_size)
        max_value = tf.cast(max_value, dtype=tf.float32)
        return max_value

    def _call_generic(self, inputs, batch_size):
        """
        Generic implementation for V, D, and J calls to avoid duplication.
        """
        dense_start, dense_end = inputs
        x = self.round_output(dense_start)
        y = self.round_output(dense_end)
        R = tf.keras.backend.greater(self.indices, x) & tf.keras.backend.less(self.indices, y)
        R = tf.cast(R, tf.float32)
        R = tf.reshape(R, shape=(batch_size, self.max_size))
        return R

    def call(self, inputs):
        """
        Dispatch to the appropriate function based on gene type.
        """
        batch_size = tf.shape(inputs[0])[0]
        return self._call_generic(inputs, batch_size)

    def compute_output_shape(self, input_shape):
        """
        Define the output shape based on the input shape.
        """
        return (None, self.max_size, 1)


class SoftCutoutLayer(Layer):
    def __init__(self, max_size, gene, **kwargs):
        super(SoftCutoutLayer, self).__init__(**kwargs)
        self.max_size = max_size
        self.gene = gene
        self.K = 5

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
        R = soft_mask(indices, x, y, self.K)
        R = tf.cast(R, tf.float32)
        R = tf.reshape(R, shape=(batch_size, self.max_size))
        return R

    def _call_d(self, inputs, batch_size):
        dense_start, dense_end = inputs
        x = self.round_output(dense_start)
        y = self.round_output(dense_end)
        indices = tf.keras.backend.arange(0, self.max_size, dtype=tf.float32)
        R = soft_mask(indices, x, y, self.K)
        R = tf.cast(R, tf.float32)
        R = tf.reshape(R, shape=(batch_size, self.max_size))
        return R

    def _call_j(self, inputs, batch_size):
        dense_start, dense_end = inputs
        x = self.round_output(dense_start)
        y = self.round_output(dense_end)
        indices = tf.keras.backend.arange(0, self.max_size, dtype=tf.float32)
        R = soft_mask(indices, x, y, self.K)
        R = tf.cast(R, tf.float32)
        R = tf.reshape(R, shape=(batch_size, self.max_size))
        return R

    def call(self, inputs):
        if self.gene == 'V':
            batch_size = tf.shape(inputs[0])[0]
            return self._call_v(inputs, batch_size)
        elif self.gene == 'D':
            batch_size = tf.shape(inputs[0])[0]
            return self._call_d(inputs, batch_size)
        elif self.gene == 'J':
            batch_size = tf.shape(inputs[0])[0]
            return self._call_j(inputs, batch_size)

    def compute_output_shape(self, input_shape):
        return (None, self.max_size, 1)


class ExtractGeneMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractGeneMask, self).__init__(**kwargs)
        self.mul_t = Multiply()  # ([input_t,mid_cu])
        self.mul_c = Multiply()  # ([input_c,mid_cu])
        self.mul_g = Multiply()  # ([input_g,mid_cu])
        self.mul_a = Multiply()  # ([input_a,mid_cu])

    def call(self, inputs):
        (a, t, g, c), mask = inputs

        masked_a = self.mul_a([a, mask])
        masked_t = self.mul_t([t, mask])
        masked_g = self.mul_g([g, mask])
        masked_c = self.mul_c([c, mask])

        return masked_a, masked_t, masked_g, masked_c


class ExtractGeneMask1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractGeneMask1D, self).__init__(**kwargs)
        self.mul_seq = Multiply()  # ([input_t,mid_cu])

    def call(self, inputs):
        seq, mask = inputs

        masked_seq = self.mul_seq([seq, mask])

        return masked_seq


class ExtractedMaskEncode(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractedMaskEncode, self).__init__(**kwargs)
        self.conv_1 = Conv1D(16, 3, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.maxpool_1 = MaxPool1D(2)
        self.conv_2 = Conv1D(32, 3, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.attention_1 = Attention()
        self.conv_3 = Conv1D(8, 8, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.maxpool_2 = MaxPool1D(2)
        self.conv_4 = Conv1D(32, 3, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.attention_2 = Attention()
        self.conv_5 = Conv1D(8, 8, padding='same', activation='relu', kernel_initializer='he_uniform')
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
        self.dense_1 = Dense(512, activation='relu')
        self.dropout_1 = Dropout(0.5)

    def call(self, inputs):
        x = self.concated(inputs)
        x = self.maxpool_1(x)
        x = self.attention([x, x])
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        return x


class Conv2D_and_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernel=(3, 3), max_pool=(2, 1), **kwargs):
        super(Conv2D_and_BatchNorm, self).__init__(**kwargs)
        self.conv_2d = Conv2D(filters, kernel, padding='same', kernel_initializer='he_uniform')  # (inputs)
        self.batch_norm = BatchNormalization()  # (concatenated_path)
        self.activation = LeakyReLU()  # (concatenated_path)
        self.max_pool = MaxPool2D(max_pool)  # (concatenated_path)

    def call(self, inputs):
        x = self.conv_2d(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class Conv1D_and_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernel=3, max_pool=2, activation=None, initializer=None, **kwargs):
        super(Conv1D_and_BatchNorm, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.max_pool_size = max_pool
        self.initializer = 'glorot_uniform' if initializer is None else initializer
        self.activation = activation if activation is not None else LeakyReLU()

    def build(self, input_shape):
        # Dynamically initialize sub-layers based on the input shape
        self.conv_2d = Conv1D(
            self.filters, self.kernel, padding='same',
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer=self.initializer
        )
        self.conv_2d_2 = Conv1D(
            self.filters, self.kernel, padding='same',
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer=self.initializer
        )
        self.conv_2d_3 = Conv1D(
            self.filters, self.kernel, padding='same',
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer=self.initializer
        )
        self.batch_norm = BatchNormalization(
            momentum=0.1, epsilon=0.8, center=True, scale=True
        )
        self.max_pool = MaxPool1D(self.max_pool_size)

        # Call parent's build method
        super(Conv1D_and_BatchNorm, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_2d(inputs)
        x = self.conv_2d_2(x)
        x = self.conv_2d_3(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x



class Conv1D_and_BatchNorm_Residual(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernel=3, max_pool=2, activation=None, initializer=None, **kwargs):
        super(Conv1D_and_BatchNorm_Residual, self).__init__(**kwargs)
        initializer_ = 'glorot_uniform' if initializer is None else initializer
        self.conv_2d_1 = Conv1D(filters, kernel, padding='same',
                                kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_)
        self.conv_2d_2 = Conv1D(filters, kernel, padding='same',
                                kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_)
        self.conv_2d_3 = Conv1D(32, kernel, padding='same',
                                kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializer_)
        self.batch_norm = BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02)
        if activation is None:
            self.activation = LeakyReLU()
        else:
            self.activation = activation
        self.max_pool = MaxPool1D(max_pool)

    def call(self, inputs):
        x = self.conv_2d_1(inputs)
        x = self.conv_2d_2(x)
        x = self.conv_2d_3(x)

        # Creating a residual connection (or skip connection)
        x = x + inputs

        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class SepConv1D_and_Residual(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernel=3, max_pool=2, activation=None, initializer=None, **kwargs):
        super(SepConv1D_and_Residual, self).__init__(**kwargs)
        initializer_ = 'glorot_uniform' if initializer is None else initializer
        self.sep_conv_2d_1 = SeparableConv1D(filters, kernel, padding='same',
                                             depthwise_regularizer=regularizers.l2(0.01),
                                             pointwise_regularizer=regularizers.l2(0.01),
                                             depthwise_initializer=initializer_,
                                             pointwise_initializer=initializer_)

        self.sep_conv_2d_2 = SeparableConv1D(filters, kernel, padding='same',
                                             depthwise_regularizer=regularizers.l2(0.01),
                                             pointwise_regularizer=regularizers.l2(0.01),
                                             depthwise_initializer=initializer_,
                                             pointwise_initializer=initializer_)

        self.sep_conv_2d_3 = SeparableConv1D(filters, kernel, padding='same',
                                             depthwise_regularizer=regularizers.l2(0.01),
                                             pointwise_regularizer=regularizers.l2(0.01),
                                             depthwise_initializer=initializer_,
                                             pointwise_initializer=initializer_)

        self.batch_norm = BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02)

        if activation is None:
            self.activation = LeakyReLU()
        else:
            self.activation = activation

        self.max_pool = MaxPool1D(max_pool)

    def call(self, inputs):
        x = self.sep_conv_2d_1(inputs)
        x = self.sep_conv_2d_2(x)
        x = self.sep_conv_2d_3(x)

        # Applying residual connection
        x = tf.keras.layers.add([x, inputs])

        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        """
        Initialize the embedding layers dynamically.
        """
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim
        )
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=self.maxlen,
            output_dim=self.embed_dim
        )

        # Call parent's build method to finalize setup
        super(TokenAndPositionEmbedding, self).build(input_shape)

    def call(self, x):
        """
        Add token embeddings and positional embeddings.
        """
        # Generate positional indices
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)  # Get positional embeddings

        x = self.token_emb(x)  # Get token embeddings
        return x + positions





class MinMaxValueConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


class SinusoidalTokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(SinusoidalTokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.emded_dim = emded_dim

        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        # Position embeddings are computed without using trainable parameters
        self.pos_emb = self.sinusoidal_pos_emb(maxlen, emded_dim)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def sinusoidal_pos_emb(self, maxlen, d_model):
        angle_rads = self.get_angles(
            np.arange(maxlen)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        x = self.token_emb(x)
        positions = self.pos_emb[:, :tf.shape(x)[1], :]
        return x + positions


class MutationOracleBody(tf.keras.layers.Layer):
    def __init__(self, activation='relu', latent_dim=1024, name='mutation_oracle_body', **kwargs):
        super(MutationOracleBody, self).__init__(**kwargs)

        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=3)
        self.flatten_before_head = Flatten()
        self.dense_before_head = Dense(latent_dim, activation=activation, name=name)
        self.dropout_before_head = Dropout(0.3)

    def call(self, inputs):
        x = self.conv_layer_1(inputs)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.flatten_before_head(x)
        x = self.dense_before_head(x)
        x = self.dropout_before_head(x)
        return x


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ActivationMonitor(Callback):
    def __init__(self, layer_name):
        super(ActivationMonitor, self).__init__()
        self.layer_name = layer_name

    def on_epoch_end(self, epoch, logs=None):
        # Get the layer output for the validation data
        layer_output = self.get_layer_output(self.validation_data[0])

        # Compute the percentage of active neurons
        active_neurons = np.mean(layer_output > 0.5)  # Assuming ReLU activation; adjust threshold if needed
        print(f"\nEpoch {epoch + 1}: {active_neurons * 100:.2f}% neurons active in {self.layer_name}")

    def get_layer_output(self, x):
        # Create a function to get the output of the desired layer
        func = tf.keras.backend.function([self.model.input], [self.model.get_layer(self.layer_name).output])
        return func([x])[0]


class SegmentationGateLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SegmentationGateLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Assuming inputs is a list [embeddings, segmentation_mask]
        embeddings, segmentation_mask = inputs

        # Ensure that the segmentation mask has the same number of dimensions as embeddings
        segmentation_mask = tf.broadcast_to(segmentation_mask, embeddings.shape)

        # Multiply embeddings with the segmentation mask
        gated_embeddings = tf.math.multiply(embeddings, segmentation_mask)

        return gated_embeddings


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, maxlen, d_model):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.maxlen = maxlen
        self.d_model = d_model

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        angle_rates = 1 / np.power(10000, (2 * (np.arange(self.d_model) // 2)) / np.float32(self.d_model))
        angle_rads = positions[:, np.newaxis] * angle_rates[np.newaxis, :]
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return x + pos_encoding


class LearnedPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, d_model):
        super(LearnedPositionalEmbedding, self).__init__()
        self.maxlen = maxlen
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class RelativePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, d_model):
        super(RelativePositionalEmbedding, self).__init__()
        self.maxlen = maxlen
        self.relative_positions = tf.keras.layers.Embedding(input_dim=2 * maxlen - 1, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions = relative_positions + self.maxlen - 1
        relative_positions = self.relative_positions(relative_positions)
        return x + relative_positions


class RealDataEvaluationCallback(Callback):
    def __init__(self, validation_data, train_dataset, filepath, period=10):
        super().__init__()
        self.validation_data = validation_data  # validation data to be used for custom evaluation
        self.train_dataset = train_dataset  # reference to the training dataset for tokenization
        self.period = period  # perform evaluation every 'period' epochs
        self.filepath = filepath  # path to save the model
        self.best_accuracy = 0.0  # to store the best accuracy achieved

        self.locus = global_genotype()
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['V']}
        self.v_alleles = sorted(list(self.v_dict))
        self.v_allele_count = len(self.v_alleles)
        self.v_allele_call_ohe = {f: i for i, f in enumerate(self.v_alleles)}
        self.v_allele_call_rev_ohe = {i: f for i, f in enumerate(self.v_alleles)}

    def dynamic_cumulative_confidence_threshold(self, prediction, percentage=0.9):
        sorted_indices = np.argsort(prediction)[::-1]
        selected_labels = []
        cumulative_confidence = 0.0

        total_confidence = sum(prediction)
        threshold = percentage * total_confidence

        for idx in sorted_indices:
            cumulative_confidence += prediction[idx]
            selected_labels.append(idx)

            if cumulative_confidence >= threshold:
                break

        return selected_labels

    def extract_prediction_alleles_dynamic_sum(self, probabilites, percentage=0.9):
        V_ratio = []
        for v_all in (probabilites):
            v_alleles = self.dynamic_cumulative_confidence_threshold(v_all, percentage=percentage)
            V_ratio.append([self.v_allele_call_rev_ohe[i] for i in v_alleles])
        return V_ratio

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:  # check if it's the right epoch
            x_val, y_val = self.validation_data
            # eval_dataset_ = self.train_dataset.tokenize_sequences(x_val)

            padded_seqs_tensor = tf.convert_to_tensor(x_val, dtype=tf.uint8)
            dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
                'tokenized_sequence': padded_seqs_tensor})

            dataset = (
                dataset_from_tensors
                .batch(512 * 20)
                .prefetch(tf.data.AUTOTUNE)
            )

            raw_predictions = []

            for i in (dataset):
                pred = self.model.predict(i, verbose=False, batch_size=64)
                for k in ['v', 'd', 'j']:
                    pred[k + '_segment'] = pred[k + '_segment'].astype(np.float16)
                raw_predictions.append(pred)

            v_segment, d_segment, j_segment, v_allele, d_allele, j_allele = [], [], [], [], [], []
            for i in raw_predictions:
                v_allele.append(i['v_allele'])
            v_allele = np.vstack(v_allele)

            V = self.extract_prediction_alleles_dynamic_sum(v_allele, percentage=0.9)
            hits = [len(set(i.split(',')) & set(j)) > 0 for i, j in zip(y_val, V)]
            accuracy = sum(hits) / len(hits)
            THH = len(list(filter(lambda x: x > 10, list(map(len, V)))))

            # Check if we have the best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

                # Construct the filename with accuracy
                filename = os.path.join(self.filepath, f"model_acc_{accuracy:.4f}")

                # Save the model in TensorFlow's SavedModel format
                self.model.save(filename, save_format='tf')  # 'tf' is optional here as it's the default format

                print(
                    f"\nEpoch {epoch + 1}: best accuracy improved to {self.best_accuracy:.4f}, saving model to {filename}")

            # Log the accuracy and THH
            logs['accuracy'] = accuracy
            logs['THH'] = THH
            print(f"\nEpoch {epoch + 1}: accuracy: {accuracy:.4f}, THH: {THH}")  # add your THH format

    #


class AlignAIRREvaluationCallback(Callback):
    def __init__(self, validation_x, validation_y, train_dataset, filepath, period=10, tokenized_x=True, monitor='v'):
        super().__init__()
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.tokenized_x = tokenized_x
        self.train_dataset = train_dataset  # reference to the training dataset for tokenization
        self.period = period  # perform evaluation every 'period' epochs
        self.filepath = filepath  # path to save the model
        self.best_accuracy = 0.0  # to store the best accuracy achieved

        self.locus = global_genotype()
        self.init_maps()
        self.monitor = monitor

        if not tokenized_x:
            self.validation_x = self.train_dataset.tokenize_sequences(validation_x)

    def init_maps(self):
        self.v_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['V']}
        self.d_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['D']}
        self.j_dict = {i.name: i.ungapped_seq.upper() for i in self.locus[0]['J']}

        self.v_alleles = sorted(list(self.v_dict))
        self.d_alleles = sorted(list(self.d_dict))
        self.d_alleles = self.d_alleles + ['Short-D']
        self.j_alleles = sorted(list(self.j_dict))

        self.v_allele_count = len(self.v_alleles)
        self.d_allele_count = len(self.d_alleles)
        self.j_allele_count = len(self.j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(self.v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(self.d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(self.j_alleles)}

        self.v_allele_call_rev_ohe = {i: f for i, f in enumerate(self.v_alleles)}
        self.d_allele_call_rev_ohe = {i: f for i, f in enumerate(self.d_alleles)}
        self.j_allele_call_rev_ohe = {i: f for i, f in enumerate(self.j_alleles)}

    def dynamic_cumulative_confidence_threshold(self, prediction, percentage=0.9):
        sorted_indices = np.argsort(prediction)[::-1]
        selected_labels = []
        cumulative_confidence = 0.0

        total_confidence = sum(prediction)
        threshold = percentage * total_confidence

        for idx in sorted_indices:
            cumulative_confidence += prediction[idx]
            selected_labels.append(idx)

            if cumulative_confidence >= threshold:
                break

        return selected_labels

    def extract_prediction_alleles_dynamic_sum(self, probabilites, percentage=0.9, type='v'):
        V_ratio = []
        for v_all in (probabilites):
            v_alleles = self.dynamic_cumulative_confidence_threshold(v_all, percentage=percentage)
            if type == 'v':
                V_ratio.append([self.v_allele_call_rev_ohe[i] for i in v_alleles])
            elif type == 'd':
                V_ratio.append([self.d_allele_call_rev_ohe[i] for i in v_alleles])
            elif type == 'j':
                V_ratio.append([self.j_allele_call_rev_ohe[i] for i in v_alleles])
        return V_ratio

    def get_processed_predictions(self, dataset):
        raw_predictions = []
        for i in dataset:
            pred = self.model.predict(i, verbose=False, batch_size=64)
            for k in ['v', 'd', 'j']:
                pred[k + '_segment'] = pred[k + '_segment'].astype(np.float16)
            raw_predictions.append(pred)
        v_allele, d_allele, j_allele = [], [], []
        mutation_rate = []
        for i in raw_predictions:
            v_allele.append(i['v_allele'])
            d_allele.append(i['d_allele'])
            j_allele.append(i['j_allele'])
            mutation_rate.append(i['mutation_rate'])
        v_allele = np.vstack(v_allele)
        d_allele = np.vstack(d_allele)
        j_allele = np.vstack(j_allele)
        mutation_rate = np.vstack(mutation_rate)
        return v_allele, d_allele, j_allele, mutation_rate

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period == 0:  # check if it's the right epoch
            x_val, y_val = self.validation_x, self.validation_y

            padded_seqs_tensor = tf.convert_to_tensor(x_val, dtype=tf.uint8)
            dataset_from_tensors = tf.data.Dataset.from_tensor_slices({
                'tokenized_sequence': padded_seqs_tensor})

            dataset = (
                dataset_from_tensors
                .batch(512 * 20)
                .prefetch(tf.data.AUTOTUNE)
            )

            v_allele, d_allele, j_allele, mutation_rate = self.get_processed_predictions(dataset)
            allele_m = {'v_call': v_allele, 'd_call': d_allele, 'j_call': j_allele}
            for call in ['v_call', 'd_call', 'j_call']:
                X = self.extract_prediction_alleles_dynamic_sum(allele_m[call], percentage=0.9, type=call.split('_')[0])
                hits = [len(set(i.split(',') if i == i else ['']) & set(j)) > 0 for i, j in zip(y_val[call], X)]
                agg = sum(hits) / len(hits)
                above_10 = len(list(filter(lambda x: x > 10, list(map(len, X)))))
                above_5 = len(list(filter(lambda x: x > 5, list(map(len, X)))))
                above_2 = len(list(filter(lambda x: x > 2, list(map(len, X)))))

                logs[call + '_agreement'] = agg
                logs[call + '_>10c'] = above_10
                logs[call + '_>5c'] = above_5
                logs[call + '_>2c'] = above_2
                if self.monitor + '_call' == call:
                    accuracy = agg
            logs['mean_mutation_rate'] = np.mean(mutation_rate)

            # Check if we have the best accuracy

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

                # Construct the filename with accuracy
                filename = os.path.join(self.filepath, f"model_acc_{accuracy:.4f}")

                # Save the model in TensorFlow's SavedModel format
                self.model.save(filename, save_format='tf')  # 'tf' is optional here as it's the default format

                print(
                    f"\nEpoch {epoch + 1}: best accuracy improved to {self.best_accuracy:.4f}, saving model to {filename}")


class Mish(Layer):
    """
    Mish Activation Function as a Layer.
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape


class AverageLastLabel(tf.keras.metrics.Mean):
    def __init__(self, name="average_last_label", **kwargs):
        super(AverageLastLabel, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get the last label of the predicted D allele if it exists
        if "d_allele" in y_pred:
            last_label = y_pred["d_allele"][:, -1]
        else:
            # If no D allele, use a default value or skip update
            # For now, we'll use zeros to maintain the metric structure
            batch_size = tf.shape(list(y_pred.values())[0])[0]
            last_label = tf.zeros((batch_size,), dtype=tf.float32)
        return super(AverageLastLabel, self).update_state(last_label, sample_weight)


class EntropyMetric(tf.keras.metrics.Mean):
    def __init__(self, allele_name, **kwargs):
        super(EntropyMetric, self).__init__(name=f"{allele_name}_entropy", **kwargs)
        self.allele_name = allele_name

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate entropy for the specified allele prediction
        entropy = -tf.reduce_sum(y_pred[self.allele_name] * tf.math.log(y_pred[self.allele_name] + 1e-9), axis=-1)
        return super(EntropyMetric, self).update_state(entropy, sample_weight)


def mod3_mse_loss(y_true, y_pred):
    y_true_casted = K.cast(y_true, dtype='float32')
    # Compute the residual of the predicted values modulo 3

    div_pred = K.cast(y_pred // 3.0, dtype='float32')
    mod_pred = y_pred - div_pred * 3

    div_true = K.cast(y_true_casted // 3.0, dtype='float32')
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
    mod_pred = K.abs(y_pred - 3 * K.cast(K.round(y_pred / 3), 'float32'))
    mod_loss = K.mean(K.square(mod_pred))
    return mse + mod_loss


def mse_no_regularization(y_true, y_pred):
    """
    Computes the mean squared error.
    """
    mse = K.mean(K.square(y_true - y_pred))
    return mse


def log_cosh_loss(y_true, y_pred):
    """
    Computes the log-cosh loss.
    """
    loss = tf.keras.losses.log_cosh(
        y_true, y_pred
    )
    return loss