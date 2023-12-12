import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, LeakyReLU, MaxPool1D, Add, Flatten, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential


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
                 conv_activation = None,initializer=None, **kwargs):
        super(ConvResidualFeatureExtractionBlock, self).__init__(**kwargs)

        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02) if initializer is None else initializer
        self.conv_activation = conv_activation
        self.conv_layers = [Conv1D_and_BatchNorm(filters=filter_size, kernel=kernel_size, max_pool=max_pool_size,
                                                 initializer=self.initializer,
                                                 activation=self.conv_activation) for _ in range(num_conv_batch_layers)]

        self.max_pool_layers = [MaxPool1D(2) for _ in range(num_conv_batch_layers)]
        self.activation_layers = [LeakyReLU() for _ in range(num_conv_batch_layers)]
        self.add_layers = [Add() for _ in range(num_conv_batch_layers)]

        self.residual_channel = Conv1D(filter_size, kernel_size, padding='same',
                                       kernel_regularizer=regularizers.l2(0.01),
                                       kernel_initializer=self.initializer)

        self.segmentation_feature_flatten = Flatten()

    def call(self, embeddings):
        # Residual
        residual_stream = self.residual_channel(embeddings)
        residual_end = self.max_pool_layers[0](residual_stream)

        feature_stream = self.conv_layers[0](embeddings)
        residual_end = self.add_layers[0]([feature_stream, residual_end])
        residual_end = self.activation_layers[0](residual_end)
        residual_end = self.max_pool_layers[0](residual_end)

        for index in range(1,len(self.max_pool_layers)):
                # get F(x)
                feature_stream = self.conv_layers[index](residual_end)
                residual_end = self.max_pool_layers[index](residual_end)
                residual_end = self.add_layers[index]([feature_stream,residual_end])
                residual_end = self.activation_layers[index](residual_end)

        residual_end = self.segmentation_feature_flatten(residual_end)
        return residual_end


