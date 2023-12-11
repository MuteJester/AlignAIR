from enum import Enum, auto
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Attention, Multiply, Reshape
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    concatenate,
    Input,
    Dropout, Add, LeakyReLU
)
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.regularizers import l1, l2

from VDeepJLayers import (
    CutoutLayer,
    mod3_mse_regularization, mse_no_regularization,
    Conv1D_and_BatchNorm,
    ExtractGeneMask1D,
    TokenAndPositionEmbedding, TransformerBlock, Conv1D, BatchNormalization, MaxPool1D,
    SinusoidalTokenAndPositionEmbedding, Mish,MinMaxValueConstraint
)

def swish(x):
    return K.sigmoid(x) * x


def mish(x):
    return x * K.tanh(K.softplus(x))


get_custom_objects().update({'mish': Activation(mish)})
class ModelComponents(Enum):
    Segmentation = auto()
    V_Classifier = auto()
    D_Classifier = auto()
    J_Classifier = auto()



class AlignAIRR(tf.keras.Model):
    """
    This is a refactored version taken from the experimental module version, this version of the architecture preforms
    the segmentation task and mutation rate estimation followed by the allele classification task.
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
    ):
        super(AlignAIRR, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)

        # Model Params
        self.max_seq_length = int(max_seq_length)
        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count


        # Hyperparams + Constants
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )


        # Tracking
        self.init_metric_trackers()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_segmentation_feature_extractor_block()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_v_feature_extraction_block()
        self._init_d_feature_extraction_block()
        self._init_j_feature_extraction_block()

        self.concatenate_input = concatenate
        self.input_embeddings = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        # Init the masking layer that will leverage the predicted segmentation mask
        self.init_masking_layers()

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()


    def init_metric_trackers(self):
        """
        here we initialize the different trackers that will automatically record model performance
        :return:
        """
        # Track the total model loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # track the intersection loss
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        # track the segmentation loss
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        # track the classification loss
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # track the mutation rate loss
        self.mutation_rate_loss_tracker = tf.keras.metrics.Mean(
            name="mutation_rate_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def init_masking_layers(self):
        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

    def _init_segmentation_feature_extractor_block(self):
        # Resnet Influenced
        self.conv_layer_segmentation_1 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_3 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_5 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)

        self.residual_connection_segmentation_conv_x_to_1 = Conv1D(64, 5, padding='same',
                                                                   kernel_regularizer=regularizers.l2(0.01),
                                                                   kernel_initializer=self.initializer)
        self.residual_connection_segmentation_max_pool_x_to_1 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_x_to_1 = LeakyReLU()
        self.residual_connection_segmentation_add_x_to_1 = Add()

        self.residual_connection_segmentation_max_pool_1_to_3 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_1_to_3 = LeakyReLU()
        self.residual_connection_segmentation_add_1_to_3 = Add()

        self.residual_connection_segmentation_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_2_to_4 = LeakyReLU()
        self.residual_connection_segmentation_add_2_to_4 = Add()

        self.residual_connection_segmentation_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_3_to_5 = LeakyReLU()
        self.residual_connection_segmentation_add_3_to_5 = Add()

        self.residual_connection_segmentation_max_pool_5_to_d = MaxPool1D(2)
        self.residual_connection_segmentation_activation_5_to_d = LeakyReLU()
        self.residual_connection_segmentation_add_5_to_d = Add()

        self.segmentation_feature_flatten = Flatten()


    def _init_v_feature_extraction_block(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_5 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_6 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

        self.residual_connection_v_features_conv_s_to_1 = Conv1D(128, 5, padding='same',
                                                                 kernel_regularizer=regularizers.l2(0.01),
                                                                 kernel_initializer=self.initializer)
        self.residual_connection_v_features_max_pool_s_to_1 = MaxPool1D(2)
        self.residual_connection_v_features_activation_s_to_1 = LeakyReLU()
        self.residual_connection_v_features_add_s_to_1 = Add()

        self.residual_connection_v_features_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_v_features_activation_2_to_4 = LeakyReLU()
        self.residual_connection_v_features_add_2_to_4 = Add()

        self.residual_connection_v_features_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_v_features_activation_3_to_5 = LeakyReLU()
        self.residual_connection_v_features_add_3_to_5 = Add()

        self.residual_connection_v_features_max_pool_4_to_6 = MaxPool1D(2)
        self.residual_connection_v_features_activation_4_to_6 = LeakyReLU()
        self.residual_connection_v_features_add_4_to_6 = Add()

        self.residual_connection_v_features_max_pool_5_to_7 = MaxPool1D(2)
        self.residual_connection_v_features_activation_5_to_7 = LeakyReLU()
        self.residual_connection_v_features_add_5_to_7 = Add()

        self.residual_connection_v_features_max_pool_6_to_f = MaxPool1D(2)
        self.residual_connection_v_features_activation_6_to_f = LeakyReLU()
        self.residual_connection_v_features_add_6_to_f = Add()

    def _init_d_feature_extraction_block(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)

        self.residual_connection_d_features_conv_s_to_1 = Conv1D(64, 5, padding='same',
                                                                 kernel_regularizer=regularizers.l2(0.01),
                                                                 kernel_initializer=self.initializer)
        self.residual_connection_d_features_max_pool_s_to_1 = MaxPool1D(2)
        self.residual_connection_d_features_activation_s_to_1 = LeakyReLU()
        self.residual_connection_d_features_add_s_to_1 = Add()

        self.residual_connection_d_features_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_d_features_activation_2_to_4 = LeakyReLU()
        self.residual_connection_d_features_add_2_to_4 = Add()

        self.residual_connection_d_features_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_d_features_activation_3_to_5 = LeakyReLU()
        self.residual_connection_d_features_add_3_to_5 = Add()

        self.residual_connection_d_features_max_pool_4_to_6 = MaxPool1D(2)
        self.residual_connection_d_features_activation_4_to_6 = LeakyReLU()
        self.residual_connection_d_features_add_4_to_6 = Add()

    def _init_j_feature_extraction_block(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

        self.residual_connection_j_features_conv_s_to_1 = Conv1D(64, 5, padding='same',
                                                                 kernel_regularizer=regularizers.l2(0.01),
                                                                 kernel_initializer=self.initializer)
        self.residual_connection_j_features_max_pool_s_to_1 = MaxPool1D(2)
        self.residual_connection_j_features_activation_s_to_1 = LeakyReLU()
        self.residual_connection_j_features_add_s_to_1 = Add()

        self.residual_connection_j_features_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_j_features_activation_2_to_4 = LeakyReLU()
        self.residual_connection_j_features_add_2_to_4 = Add()

        self.residual_connection_j_features_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_j_features_activation_3_to_5 = LeakyReLU()
        self.residual_connection_j_features_add_3_to_5 = Add()

        self.residual_connection_j_features_max_pool_4_to_6 = MaxPool1D(2)
        self.residual_connection_j_features_activation_4_to_6 = LeakyReLU()
        self.residual_connection_j_features_add_4_to_6 = Add()

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle", kernel_initializer=self.initializer,
            )

        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="sigmoid", name="v_allele"#,kernel_regularizer=regularizers.l1(0.01)
        )

    def _init_j_classification_layers(self):

        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele",kernel_regularizer=regularizers.l1(0.01)
        )

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele",kernel_regularizer=regularizers.l1(0.01)
        )

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

        self.mutation_rate_mid = Dense(
            self.max_seq_length//2, activation=act, name="mutation_rate_mid", kernel_initializer=self.initializer
        )
        self.mutation_rate_dropout = Dropout(0.05)
        self.mutation_rate_head = Dense(
            1, activation='relu', name="mutation_rate", kernel_initializer=self.initializer
            ,kernel_constraint=MinMaxValueConstraint(0, 1)
        )

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def predict_segments(self, concatenated_signals):
        v_segment_mid = self.v_segment_mid(concatenated_signals)
        v_segment = self.v_segment_out(v_segment_mid)

        d_segment_mid = self.d_segment_mid(concatenated_signals)
        d_segment = self.d_segment_out(d_segment_mid)

        j_segment_mid = self.j_segment_mid(concatenated_signals)
        j_segment = self.j_segment_out(j_segment_mid)


        mutation_rate_mid = self.mutation_rate_mid(concatenated_signals)
        mutation_rate_mid = self.mutation_rate_dropout(mutation_rate_mid)
        mutation_rate = self.mutation_rate_head(mutation_rate_mid)

        return v_segment, d_segment, j_segment, mutation_rate

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_allele_middle = self.v_allele_mid(v_feature_map)
        v_allele = self.v_allele_call_head(v_allele_middle)

        # ============================ D =============================
        d_allele_middle = self.d_allele_mid(d_feature_map)
        d_allele = self.d_allele_call_head(d_allele_middle)

        # ============================ J =============================
        j_allele_middle = self.j_allele_mid(j_feature_map)
        j_allele = self.j_allele_call_head(j_allele_middle)

        return v_allele, d_allele, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):

        s = self.residual_connection_v_features_conv_s_to_1(concatenated_v_mask_input_embedding)

        # residual 1
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_residual_1 = self.residual_connection_v_features_max_pool_s_to_1(s)
        v_residual_1 = self.residual_connection_v_features_add_s_to_1([v_residual_1, v_conv_layer_1])
        v_residual_1 = self.residual_connection_v_features_activation_s_to_1(v_residual_1)

        v_conv_layer_2 = self.conv_v_layer_2(v_residual_1)

        # residual 2
        v_residual_2 = self.residual_connection_v_features_max_pool_2_to_4(v_residual_1)
        v_residual_2 = self.residual_connection_v_features_add_2_to_4([v_residual_2, v_conv_layer_2])
        v_residual_2 = self.residual_connection_v_features_activation_2_to_4(v_residual_2)

        v_conv_layer_3 = self.conv_v_layer_3(v_residual_2)

        # residual 3
        v_residual_3 = self.residual_connection_v_features_max_pool_3_to_5(v_residual_2)
        v_residual_3 = self.residual_connection_v_features_add_3_to_5([v_residual_3, v_conv_layer_3])
        v_residual_3 = self.residual_connection_v_features_activation_3_to_5(v_residual_3)

        v_conv_layer_4 = self.conv_v_layer_4(v_residual_3)

        # residual 4
        v_residual_4 = self.residual_connection_v_features_max_pool_4_to_6(v_residual_3)
        v_residual_4 = self.residual_connection_v_features_add_4_to_6([v_residual_4, v_conv_layer_4])
        v_residual_4 = self.residual_connection_v_features_activation_4_to_6(v_residual_4)

        v_conv_layer_5 = self.conv_v_layer_5(v_residual_4)

        # residual 5
        v_residual_5 = self.residual_connection_v_features_max_pool_5_to_7(v_residual_4)
        v_residual_5 = self.residual_connection_v_features_add_5_to_7([v_residual_5, v_conv_layer_5])
        v_residual_5 = self.residual_connection_v_features_activation_5_to_7(v_residual_5)

        v_conv_layer_6 = self.conv_v_layer_6(v_residual_5)

        # residual 6
        v_residual_6 = self.residual_connection_v_features_max_pool_6_to_f(v_residual_5)
        v_residual_6 = self.residual_connection_v_features_add_6_to_f([v_residual_6, v_conv_layer_6])
        v_residual_6 = self.residual_connection_v_features_activation_6_to_f(v_residual_6)

        v_feature_map = Flatten()(v_residual_6)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):

        s = self.residual_connection_d_features_conv_s_to_1(concatenated_d_mask_input_embedding)

        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        # residual 1
        d_residual_1 = self.residual_connection_d_features_max_pool_s_to_1(s)
        d_residual_1 = self.residual_connection_d_features_add_s_to_1([d_residual_1, d_conv_layer_1])
        d_residual_1 = self.residual_connection_d_features_activation_s_to_1(d_residual_1)

        d_conv_layer_2 = self.conv_d_layer_2(d_residual_1)
        # residual 2
        d_residual_2 = self.residual_connection_d_features_max_pool_2_to_4(d_residual_1)
        d_residual_2 = self.residual_connection_d_features_add_2_to_4([d_residual_2, d_conv_layer_2])
        d_residual_2 = self.residual_connection_d_features_activation_2_to_4(d_residual_2)

        d_conv_layer_3 = self.conv_d_layer_3(d_residual_2)
        # residual 3
        d_residual_3 = self.residual_connection_d_features_max_pool_3_to_5(d_residual_2)
        d_residual_3 = self.residual_connection_d_features_add_3_to_5([d_residual_3, d_conv_layer_3])
        d_residual_3 = self.residual_connection_d_features_activation_3_to_5(d_residual_3)

        d_feature_map = self.conv_d_layer_4(d_residual_3)
        # residual 4
        d_residual_4 = self.residual_connection_d_features_max_pool_4_to_6(d_residual_3)
        d_residual_4 = self.residual_connection_d_features_add_4_to_6([d_residual_4, d_feature_map])
        d_residual_4 = self.residual_connection_d_features_activation_4_to_6(d_residual_4)

        d_feature_map = Flatten()(d_residual_4)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        s = self.residual_connection_j_features_conv_s_to_1(concatenated_j_mask_input_embedding)

        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        # residual 1
        j_residual_1 = self.residual_connection_j_features_max_pool_s_to_1(s)
        j_residual_1 = self.residual_connection_j_features_add_s_to_1([j_residual_1, j_conv_layer_1])
        j_residual_1 = self.residual_connection_j_features_activation_s_to_1(j_residual_1)

        j_conv_layer_2 = self.conv_j_layer_2(j_residual_1)
        # residual 2
        j_residual_2 = self.residual_connection_j_features_max_pool_2_to_4(j_residual_1)
        j_residual_2 = self.residual_connection_j_features_add_2_to_4([j_residual_2, j_conv_layer_2])
        j_residual_2 = self.residual_connection_j_features_activation_2_to_4(j_residual_2)

        j_conv_layer_3 = self.conv_j_layer_3(j_residual_2)
        # residual 3
        j_residual_3 = self.residual_connection_j_features_max_pool_3_to_5(j_residual_2)
        j_residual_3 = self.residual_connection_j_features_add_3_to_5([j_residual_3, j_conv_layer_3])
        j_residual_3 = self.residual_connection_j_features_activation_3_to_5(j_residual_3)

        j_feature_map = self.conv_j_layer_4(j_residual_3)
        # residual 4
        j_residual_4 = self.residual_connection_j_features_max_pool_4_to_6(j_residual_3)
        j_residual_4 = self.residual_connection_j_features_add_4_to_6([j_residual_4, j_feature_map])
        j_residual_4 = self.residual_connection_j_features_activation_4_to_6(j_residual_4)

        j_feature_map = Flatten()(j_residual_4)
        return j_feature_map

    def _forward_pass_segmentation_feature_extraction(self,concatenated_input_embedding):
        # Residual
        residual_connection_segmentation_conv = self.residual_connection_segmentation_conv_x_to_1(
            concatenated_input_embedding)
        residual_connection_segmentation_max_pool = self.residual_connection_segmentation_max_pool_x_to_1(
            residual_connection_segmentation_conv)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        conv_layer_segmentation_1 = self.conv_layer_segmentation_1(concatenated_input_embedding)
        conv_layer_segmentation_1_res = self.residual_connection_segmentation_add_x_to_1(
            [conv_layer_segmentation_1, residual_connection_segmentation_max_pool])
        conv_layer_segmentation_1_res = self.residual_connection_segmentation_activation_x_to_1(
            conv_layer_segmentation_1_res)

        conv_layer_segmentation_2 = self.conv_layer_segmentation_2(conv_layer_segmentation_1_res)

        # residual 2
        conv_layer_segmentation_2_res = self.residual_connection_segmentation_max_pool_1_to_3(
            conv_layer_segmentation_1_res)
        conv_layer_segmentation_2_res = self.residual_connection_segmentation_add_1_to_3(
            [conv_layer_segmentation_2_res, conv_layer_segmentation_2])
        conv_layer_segmentation_2_res = self.residual_connection_segmentation_activation_1_to_3(
            conv_layer_segmentation_2_res)

        conv_layer_segmentation_3 = self.conv_layer_segmentation_3(conv_layer_segmentation_2_res)

        # residual 3
        conv_layer_segmentation_3_res = self.residual_connection_segmentation_max_pool_2_to_4(
            conv_layer_segmentation_2_res)
        conv_layer_segmentation_3_res = self.residual_connection_segmentation_add_2_to_4(
            [conv_layer_segmentation_3_res, conv_layer_segmentation_3])
        conv_layer_segmentation_3_res = self.residual_connection_segmentation_activation_2_to_4(
            conv_layer_segmentation_3_res)

        conv_layer_segmentation_4 = self.conv_layer_segmentation_4(conv_layer_segmentation_3_res)

        # residual 4
        conv_layer_segmentation_5_res = self.residual_connection_segmentation_max_pool_3_to_5(
            conv_layer_segmentation_3_res)
        conv_layer_segmentation_5_res = self.residual_connection_segmentation_add_3_to_5(
            [conv_layer_segmentation_5_res, conv_layer_segmentation_4])
        conv_layer_segmentation_5_res = self.residual_connection_segmentation_activation_3_to_5(
            conv_layer_segmentation_5_res)

        last_conv_layer = self.conv_layer_segmentation_5(conv_layer_segmentation_5_res)

        # residual 5
        conv_layer_segmentation_d_res = self.residual_connection_segmentation_max_pool_5_to_d(
            conv_layer_segmentation_5_res)
        conv_layer_segmentation_d_res = self.residual_connection_segmentation_add_5_to_d(
            [conv_layer_segmentation_d_res, last_conv_layer])
        conv_layer_segmentation_d_res = self.residual_connection_segmentation_activation_5_to_d(
            conv_layer_segmentation_d_res)

        return conv_layer_segmentation_d_res

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        conv_layer_segmentation_d_res = self._forward_pass_segmentation_feature_extraction(input_embeddings)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, input_embeddings])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, input_embeddings])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, input_embeddings])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(masked_sequence_v)
        d_feature_map = self._encode_masked_d_signal(masked_sequence_d)
        j_feature_map = self._encode_masked_j_signal(masked_sequence_j)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
            'mutation_rate': mutation_rate
        }

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def d_loss(self,y_true, y_pred, penalty_factor=1.0, last_label_penalty_factor=1.0):
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
        other_labels_above_zero = K.cast(K.any(K.greater(y_pred[:, :-1], 0), axis=1), tf.float32)  # Check if any other label > 0
        last_label_penalty = last_label_penalty_factor * last_label_high_confidence * other_labels_above_zero


        # Combined loss with both penalties
        return bce + extra_penalty + last_label_penalty

    def multi_task_loss(self, y_true, y_pred):
        # Extract the segmentation and classification outputs
        segmentation_true = [self.c2f32(y_true[k]) for k in ['v_segment', 'd_segment', 'j_segment']]
        segmentation_pred = [self.c2f32(y_pred[k]) for k in ['v_segment', 'd_segment', 'j_segment']]

        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        # Compute the segmentation loss
        v_segment_loss = tf.keras.metrics.binary_crossentropy(segmentation_true[0], segmentation_pred[0])
        d_segment_loss = tf.keras.metrics.binary_crossentropy(segmentation_true[1], segmentation_pred[1])
        j_segment_loss = tf.keras.metrics.binary_crossentropy(segmentation_true[2], segmentation_pred[2])

        total_segmentation_loss = v_segment_loss + d_segment_loss + j_segment_loss

        # Compute the intersection loss
        v_d_intersection = K.sum(segmentation_pred[0] * segmentation_pred[1])
        v_j_intersection = K.sum(segmentation_pred[0] * segmentation_pred[2])
        d_j_intersection = K.sum(segmentation_pred[1] * segmentation_pred[2])

        total_intersection_loss = v_d_intersection + v_j_intersection + d_j_intersection

        # Compute the classification loss
        clf_v_loss = tf.keras.metrics.binary_crossentropy(classification_true[0], classification_pred[0])
        # origianl
        #clf_d_loss = tf.keras.metrics.binary_crossentropy(classification_true[1], classification_pred[1])
        clf_d_loss = self.d_loss(classification_true[1], classification_pred[1],penalty_factor=1,last_label_penalty_factor=3)

        clf_j_loss = tf.keras.metrics.binary_crossentropy(classification_true[2], classification_pred[2])

        mutation_rate_loss = tf.keras.metrics.mean_absolute_error(self.c2f32(y_true['mutation_rate']),
                                                                  self.c2f32(y_pred['mutation_rate']))

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # Combine the losses using a weighted sum
        total_loss = (
                self.segmentation_weight * total_segmentation_loss
                + self.intersection_weight * total_intersection_loss
                + self.classification_weight * classification_loss
                + mutation_rate_loss

        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss
            ) = self.multi_task_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(total_loss)
        self.intersection_loss_tracker.update_state(total_intersection_loss)
        self.total_segmentation_loss_tracker.update_state(total_segmentation_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.mutation_rate_loss_tracker.update_state(mutation_rate_loss)

        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()
        metrics["mutation_rate_loss"] = self.mutation_rate_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.input_embeddings,
            self.conv_layer_1,
            self.conv_layer_2,
            self.conv_layer_3,
            self.conv_layer_4,
            self.v_start_mid,
            self.v_start_out,
            self.v_end_mid,
            self.v_end_out,
            self.d_start_mid,
            self.d_start_out,
            self.d_end_mid,
            self.d_end_out,
            self.j_start_mid,
            self.j_start_out,
            self.j_end_mid,
            self.j_end_out,
        ]:
            layer.trainable = False

    def freeze_component(self, component):
        if component == ModelComponents.Segmentation:
            self._freeze_segmentation_component()
        elif component == ModelComponents.V_Classifier:
            self._freeze_v_classifier_component()
        elif component == ModelComponents.D_Classifier:
            self._freeze_d_classifier_component()
        elif component == ModelComponents.J_Classifier:
            self._freeze_j_classifier_component()

    def model_summary(self, input_shape):
        x = {
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }

        return Model(inputs=x, outputs=self.call(x)).summary()

    def plot_model(self, input_shape, show_shapes=True):
        x = {
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )



class AlignAIRR_Light(tf.keras.Model):
    """
    This is a refactored version taken from the experimental module version, this version of the architecture preforms
    the segmentation task and mutation rate estimation followed by the allele classification task.

    The light version is less memory hungry and is suited for use cases when one wants quickly to predict only
    the labels for the V,D and J genes and the mutation rate, the memory heavy segmentation masks will not be
    accumulated and returned.
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
    ):
        super(AlignAIRR_Light, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)

        # Model Params
        self.max_seq_length = int(max_seq_length)
        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count


        # Hyperparams + Constants
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )


        # Tracking
        self.init_metric_trackers()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_segmentation_feature_extractor_block()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_v_feature_extraction_block()
        self._init_d_feature_extraction_block()
        self._init_j_feature_extraction_block()

        self.concatenate_input = concatenate
        self.input_embeddings = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        # Init the masking layer that will leverage the predicted segmentation mask
        self.init_masking_layers()

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()


    def init_metric_trackers(self):
        """
        here we initialize the different trackers that will automatically record model performance
        :return:
        """
        # Track the total model loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # track the intersection loss
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        # track the segmentation loss
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        # track the classification loss
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # track the mutation rate loss
        self.mutation_rate_loss_tracker = tf.keras.metrics.Mean(
            name="mutation_rate_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def init_masking_layers(self):
        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

    def _init_segmentation_feature_extractor_block(self):
        # Resnet Influenced
        self.conv_layer_segmentation_1 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_3 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_5 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)

        self.residual_connection_segmentation_conv_x_to_1 = Conv1D(64, 5, padding='same',
                                                                   kernel_regularizer=regularizers.l2(0.01),
                                                                   kernel_initializer=self.initializer)
        self.residual_connection_segmentation_max_pool_x_to_1 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_x_to_1 = LeakyReLU()
        self.residual_connection_segmentation_add_x_to_1 = Add()

        self.residual_connection_segmentation_max_pool_1_to_3 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_1_to_3 = LeakyReLU()
        self.residual_connection_segmentation_add_1_to_3 = Add()

        self.residual_connection_segmentation_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_2_to_4 = LeakyReLU()
        self.residual_connection_segmentation_add_2_to_4 = Add()

        self.residual_connection_segmentation_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_3_to_5 = LeakyReLU()
        self.residual_connection_segmentation_add_3_to_5 = Add()

        self.residual_connection_segmentation_max_pool_5_to_d = MaxPool1D(2)
        self.residual_connection_segmentation_activation_5_to_d = LeakyReLU()
        self.residual_connection_segmentation_add_5_to_d = Add()

        self.segmentation_feature_flatten = Flatten()


    def _init_v_feature_extraction_block(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_5 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_6 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

        self.residual_connection_v_features_conv_s_to_1 = Conv1D(128, 5, padding='same',
                                                                 kernel_regularizer=regularizers.l2(0.01),
                                                                 kernel_initializer=self.initializer)
        self.residual_connection_v_features_max_pool_s_to_1 = MaxPool1D(2)
        self.residual_connection_v_features_activation_s_to_1 = LeakyReLU()
        self.residual_connection_v_features_add_s_to_1 = Add()

        self.residual_connection_v_features_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_v_features_activation_2_to_4 = LeakyReLU()
        self.residual_connection_v_features_add_2_to_4 = Add()

        self.residual_connection_v_features_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_v_features_activation_3_to_5 = LeakyReLU()
        self.residual_connection_v_features_add_3_to_5 = Add()

        self.residual_connection_v_features_max_pool_4_to_6 = MaxPool1D(2)
        self.residual_connection_v_features_activation_4_to_6 = LeakyReLU()
        self.residual_connection_v_features_add_4_to_6 = Add()

        self.residual_connection_v_features_max_pool_5_to_7 = MaxPool1D(2)
        self.residual_connection_v_features_activation_5_to_7 = LeakyReLU()
        self.residual_connection_v_features_add_5_to_7 = Add()

        self.residual_connection_v_features_max_pool_6_to_f = MaxPool1D(2)
        self.residual_connection_v_features_activation_6_to_f = LeakyReLU()
        self.residual_connection_v_features_add_6_to_f = Add()

    def _init_d_feature_extraction_block(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)

        self.residual_connection_d_features_conv_s_to_1 = Conv1D(64, 5, padding='same',
                                                                 kernel_regularizer=regularizers.l2(0.01),
                                                                 kernel_initializer=self.initializer)
        self.residual_connection_d_features_max_pool_s_to_1 = MaxPool1D(2)
        self.residual_connection_d_features_activation_s_to_1 = LeakyReLU()
        self.residual_connection_d_features_add_s_to_1 = Add()

        self.residual_connection_d_features_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_d_features_activation_2_to_4 = LeakyReLU()
        self.residual_connection_d_features_add_2_to_4 = Add()

        self.residual_connection_d_features_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_d_features_activation_3_to_5 = LeakyReLU()
        self.residual_connection_d_features_add_3_to_5 = Add()

        self.residual_connection_d_features_max_pool_4_to_6 = MaxPool1D(2)
        self.residual_connection_d_features_activation_4_to_6 = LeakyReLU()
        self.residual_connection_d_features_add_4_to_6 = Add()

    def _init_j_feature_extraction_block(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

        self.residual_connection_j_features_conv_s_to_1 = Conv1D(64, 5, padding='same',
                                                                 kernel_regularizer=regularizers.l2(0.01),
                                                                 kernel_initializer=self.initializer)
        self.residual_connection_j_features_max_pool_s_to_1 = MaxPool1D(2)
        self.residual_connection_j_features_activation_s_to_1 = LeakyReLU()
        self.residual_connection_j_features_add_s_to_1 = Add()

        self.residual_connection_j_features_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_j_features_activation_2_to_4 = LeakyReLU()
        self.residual_connection_j_features_add_2_to_4 = Add()

        self.residual_connection_j_features_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_j_features_activation_3_to_5 = LeakyReLU()
        self.residual_connection_j_features_add_3_to_5 = Add()

        self.residual_connection_j_features_max_pool_4_to_6 = MaxPool1D(2)
        self.residual_connection_j_features_activation_4_to_6 = LeakyReLU()
        self.residual_connection_j_features_add_4_to_6 = Add()

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle", kernel_initializer=self.initializer,
            )

        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="sigmoid", name="v_allele"
        )

    def _init_j_classification_layers(self):

        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele"
        )

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele"
        )

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

        self.mutation_rate_mid = Dense(
            self.max_seq_length//2, activation=act, name="mutation_rate_mid", kernel_initializer=self.initializer
        )
        self.mutation_rate_dropout = Dropout(0.05)
        self.mutation_rate_head = Dense(
            1, activation='relu', name="mutation_rate", kernel_initializer=self.initializer
            ,kernel_constraint=MinMaxValueConstraint(0, 1)
        )

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def predict_segments(self, concatenated_signals):
        v_segment_mid = self.v_segment_mid(concatenated_signals)
        v_segment = self.v_segment_out(v_segment_mid)

        d_segment_mid = self.d_segment_mid(concatenated_signals)
        d_segment = self.d_segment_out(d_segment_mid)

        j_segment_mid = self.j_segment_mid(concatenated_signals)
        j_segment = self.j_segment_out(j_segment_mid)


        mutation_rate_mid = self.mutation_rate_mid(concatenated_signals)
        mutation_rate_mid = self.mutation_rate_dropout(mutation_rate_mid)
        mutation_rate = self.mutation_rate_head(mutation_rate_mid)

        return v_segment, d_segment, j_segment, mutation_rate

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_allele_middle = self.v_allele_mid(v_feature_map)
        v_allele = self.v_allele_call_head(v_allele_middle)

        # ============================ D =============================
        d_allele_middle = self.d_allele_mid(d_feature_map)
        d_allele = self.d_allele_call_head(d_allele_middle)

        # ============================ J =============================
        j_allele_middle = self.j_allele_mid(j_feature_map)
        j_allele = self.j_allele_call_head(j_allele_middle)

        return v_allele, d_allele, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):

        s = self.residual_connection_v_features_conv_s_to_1(concatenated_v_mask_input_embedding)

        # residual 1
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_residual_1 = self.residual_connection_v_features_max_pool_s_to_1(s)
        v_residual_1 = self.residual_connection_v_features_add_s_to_1([v_residual_1, v_conv_layer_1])
        v_residual_1 = self.residual_connection_v_features_activation_s_to_1(v_residual_1)

        v_conv_layer_2 = self.conv_v_layer_2(v_residual_1)

        # residual 2
        v_residual_2 = self.residual_connection_v_features_max_pool_2_to_4(v_residual_1)
        v_residual_2 = self.residual_connection_v_features_add_2_to_4([v_residual_2, v_conv_layer_2])
        v_residual_2 = self.residual_connection_v_features_activation_2_to_4(v_residual_2)

        v_conv_layer_3 = self.conv_v_layer_3(v_residual_2)

        # residual 3
        v_residual_3 = self.residual_connection_v_features_max_pool_3_to_5(v_residual_2)
        v_residual_3 = self.residual_connection_v_features_add_3_to_5([v_residual_3, v_conv_layer_3])
        v_residual_3 = self.residual_connection_v_features_activation_3_to_5(v_residual_3)

        v_conv_layer_4 = self.conv_v_layer_4(v_residual_3)

        # residual 4
        v_residual_4 = self.residual_connection_v_features_max_pool_4_to_6(v_residual_3)
        v_residual_4 = self.residual_connection_v_features_add_4_to_6([v_residual_4, v_conv_layer_4])
        v_residual_4 = self.residual_connection_v_features_activation_4_to_6(v_residual_4)

        v_conv_layer_5 = self.conv_v_layer_5(v_residual_4)

        # residual 5
        v_residual_5 = self.residual_connection_v_features_max_pool_5_to_7(v_residual_4)
        v_residual_5 = self.residual_connection_v_features_add_5_to_7([v_residual_5, v_conv_layer_5])
        v_residual_5 = self.residual_connection_v_features_activation_5_to_7(v_residual_5)

        v_conv_layer_6 = self.conv_v_layer_6(v_residual_5)

        # residual 6
        v_residual_6 = self.residual_connection_v_features_max_pool_6_to_f(v_residual_5)
        v_residual_6 = self.residual_connection_v_features_add_6_to_f([v_residual_6, v_conv_layer_6])
        v_residual_6 = self.residual_connection_v_features_activation_6_to_f(v_residual_6)

        v_feature_map = Flatten()(v_residual_6)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):

        s = self.residual_connection_d_features_conv_s_to_1(concatenated_d_mask_input_embedding)

        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        # residual 1
        d_residual_1 = self.residual_connection_d_features_max_pool_s_to_1(s)
        d_residual_1 = self.residual_connection_d_features_add_s_to_1([d_residual_1, d_conv_layer_1])
        d_residual_1 = self.residual_connection_d_features_activation_s_to_1(d_residual_1)

        d_conv_layer_2 = self.conv_d_layer_2(d_residual_1)
        # residual 2
        d_residual_2 = self.residual_connection_d_features_max_pool_2_to_4(d_residual_1)
        d_residual_2 = self.residual_connection_d_features_add_2_to_4([d_residual_2, d_conv_layer_2])
        d_residual_2 = self.residual_connection_d_features_activation_2_to_4(d_residual_2)

        d_conv_layer_3 = self.conv_d_layer_3(d_residual_2)
        # residual 3
        d_residual_3 = self.residual_connection_d_features_max_pool_3_to_5(d_residual_2)
        d_residual_3 = self.residual_connection_d_features_add_3_to_5([d_residual_3, d_conv_layer_3])
        d_residual_3 = self.residual_connection_d_features_activation_3_to_5(d_residual_3)

        d_feature_map = self.conv_d_layer_4(d_residual_3)
        # residual 4
        d_residual_4 = self.residual_connection_d_features_max_pool_4_to_6(d_residual_3)
        d_residual_4 = self.residual_connection_d_features_add_4_to_6([d_residual_4, d_feature_map])
        d_residual_4 = self.residual_connection_d_features_activation_4_to_6(d_residual_4)

        d_feature_map = Flatten()(d_residual_4)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        s = self.residual_connection_j_features_conv_s_to_1(concatenated_j_mask_input_embedding)

        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        # residual 1
        j_residual_1 = self.residual_connection_j_features_max_pool_s_to_1(s)
        j_residual_1 = self.residual_connection_j_features_add_s_to_1([j_residual_1, j_conv_layer_1])
        j_residual_1 = self.residual_connection_j_features_activation_s_to_1(j_residual_1)

        j_conv_layer_2 = self.conv_j_layer_2(j_residual_1)
        # residual 2
        j_residual_2 = self.residual_connection_j_features_max_pool_2_to_4(j_residual_1)
        j_residual_2 = self.residual_connection_j_features_add_2_to_4([j_residual_2, j_conv_layer_2])
        j_residual_2 = self.residual_connection_j_features_activation_2_to_4(j_residual_2)

        j_conv_layer_3 = self.conv_j_layer_3(j_residual_2)
        # residual 3
        j_residual_3 = self.residual_connection_j_features_max_pool_3_to_5(j_residual_2)
        j_residual_3 = self.residual_connection_j_features_add_3_to_5([j_residual_3, j_conv_layer_3])
        j_residual_3 = self.residual_connection_j_features_activation_3_to_5(j_residual_3)

        j_feature_map = self.conv_j_layer_4(j_residual_3)
        # residual 4
        j_residual_4 = self.residual_connection_j_features_max_pool_4_to_6(j_residual_3)
        j_residual_4 = self.residual_connection_j_features_add_4_to_6([j_residual_4, j_feature_map])
        j_residual_4 = self.residual_connection_j_features_activation_4_to_6(j_residual_4)

        j_feature_map = Flatten()(j_residual_4)
        return j_feature_map

    def _forward_pass_segmentation_feature_extraction(self,concatenated_input_embedding):
        # Residual
        residual_connection_segmentation_conv = self.residual_connection_segmentation_conv_x_to_1(
            concatenated_input_embedding)
        residual_connection_segmentation_max_pool = self.residual_connection_segmentation_max_pool_x_to_1(
            residual_connection_segmentation_conv)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        conv_layer_segmentation_1 = self.conv_layer_segmentation_1(concatenated_input_embedding)
        conv_layer_segmentation_1_res = self.residual_connection_segmentation_add_x_to_1(
            [conv_layer_segmentation_1, residual_connection_segmentation_max_pool])
        conv_layer_segmentation_1_res = self.residual_connection_segmentation_activation_x_to_1(
            conv_layer_segmentation_1_res)

        conv_layer_segmentation_2 = self.conv_layer_segmentation_2(conv_layer_segmentation_1_res)

        # residual 2
        conv_layer_segmentation_2_res = self.residual_connection_segmentation_max_pool_1_to_3(
            conv_layer_segmentation_1_res)
        conv_layer_segmentation_2_res = self.residual_connection_segmentation_add_1_to_3(
            [conv_layer_segmentation_2_res, conv_layer_segmentation_2])
        conv_layer_segmentation_2_res = self.residual_connection_segmentation_activation_1_to_3(
            conv_layer_segmentation_2_res)

        conv_layer_segmentation_3 = self.conv_layer_segmentation_3(conv_layer_segmentation_2_res)

        # residual 3
        conv_layer_segmentation_3_res = self.residual_connection_segmentation_max_pool_2_to_4(
            conv_layer_segmentation_2_res)
        conv_layer_segmentation_3_res = self.residual_connection_segmentation_add_2_to_4(
            [conv_layer_segmentation_3_res, conv_layer_segmentation_3])
        conv_layer_segmentation_3_res = self.residual_connection_segmentation_activation_2_to_4(
            conv_layer_segmentation_3_res)

        conv_layer_segmentation_4 = self.conv_layer_segmentation_4(conv_layer_segmentation_3_res)

        # residual 4
        conv_layer_segmentation_5_res = self.residual_connection_segmentation_max_pool_3_to_5(
            conv_layer_segmentation_3_res)
        conv_layer_segmentation_5_res = self.residual_connection_segmentation_add_3_to_5(
            [conv_layer_segmentation_5_res, conv_layer_segmentation_4])
        conv_layer_segmentation_5_res = self.residual_connection_segmentation_activation_3_to_5(
            conv_layer_segmentation_5_res)

        last_conv_layer = self.conv_layer_segmentation_5(conv_layer_segmentation_5_res)

        # residual 5
        conv_layer_segmentation_d_res = self.residual_connection_segmentation_max_pool_5_to_d(
            conv_layer_segmentation_5_res)
        conv_layer_segmentation_d_res = self.residual_connection_segmentation_add_5_to_d(
            [conv_layer_segmentation_d_res, last_conv_layer])
        conv_layer_segmentation_d_res = self.residual_connection_segmentation_activation_5_to_d(
            conv_layer_segmentation_d_res)

        return conv_layer_segmentation_d_res

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        conv_layer_segmentation_d_res = self._forward_pass_segmentation_feature_extraction(input_embeddings)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, input_embeddings])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, input_embeddings])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, input_embeddings])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(masked_sequence_v)
        d_feature_map = self._encode_masked_d_signal(masked_sequence_d)
        j_feature_map = self._encode_masked_j_signal(masked_sequence_j)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
            'mutation_rate': mutation_rate
        }

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def multi_task_loss(self, y_true, y_pred):
        # Extract the segmentation and classification outputs
        segmentation_true = [self.c2f32(y_true[k]) for k in ['v_segment', 'd_segment', 'j_segment']]
        segmentation_pred = [self.c2f32(y_pred[k]) for k in ['v_segment', 'd_segment', 'j_segment']]

        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        # Compute the segmentation loss
        v_segment_loss = tf.keras.metrics.binary_crossentropy(segmentation_true[0], segmentation_pred[0])
        d_segment_loss = tf.keras.metrics.binary_crossentropy(segmentation_true[1], segmentation_pred[1])
        j_segment_loss = tf.keras.metrics.binary_crossentropy(segmentation_true[2], segmentation_pred[2])

        total_segmentation_loss = v_segment_loss + d_segment_loss + j_segment_loss

        # Compute the intersection loss
        v_d_intersection = K.sum(segmentation_pred[0] * segmentation_pred[1])
        v_j_intersection = K.sum(segmentation_pred[0] * segmentation_pred[2])
        d_j_intersection = K.sum(segmentation_pred[1] * segmentation_pred[2])

        total_intersection_loss = v_d_intersection + v_j_intersection + d_j_intersection

        # Compute the classification loss
        clf_v_loss = tf.keras.metrics.binary_crossentropy(classification_true[0], classification_pred[0])
        clf_d_loss = tf.keras.metrics.binary_crossentropy(classification_true[1], classification_pred[1])
        clf_j_loss = tf.keras.metrics.binary_crossentropy(classification_true[2], classification_pred[2])

        mutation_rate_loss = tf.keras.metrics.mean_absolute_error(self.c2f32(y_true['mutation_rate']),
                                                                  self.c2f32(y_pred['mutation_rate']))

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # Combine the losses using a weighted sum
        total_loss = (
                self.segmentation_weight * total_segmentation_loss
                + self.intersection_weight * total_intersection_loss
                + self.classification_weight * classification_loss
                + mutation_rate_loss

        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss
            ) = self.multi_task_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(total_loss)
        self.intersection_loss_tracker.update_state(total_intersection_loss)
        self.total_segmentation_loss_tracker.update_state(total_segmentation_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.mutation_rate_loss_tracker.update_state(mutation_rate_loss)

        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()
        metrics["mutation_rate_loss"] = self.mutation_rate_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.input_embeddings,
            self.conv_layer_1,
            self.conv_layer_2,
            self.conv_layer_3,
            self.conv_layer_4,
            self.v_start_mid,
            self.v_start_out,
            self.v_end_mid,
            self.v_end_out,
            self.d_start_mid,
            self.d_start_out,
            self.d_end_mid,
            self.d_end_out,
            self.j_start_mid,
            self.j_start_out,
            self.j_end_mid,
            self.j_end_out,
        ]:
            layer.trainable = False

    def freeze_component(self, component):
        if component == ModelComponents.Segmentation:
            self._freeze_segmentation_component()
        elif component == ModelComponents.V_Classifier:
            self._freeze_v_classifier_component()
        elif component == ModelComponents.D_Classifier:
            self._freeze_d_classifier_component()
        elif component == ModelComponents.J_Classifier:
            self._freeze_j_classifier_component()

    def model_summary(self, input_shape):
        x = {
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }

        return Model(inputs=x, outputs=self.call(x)).summary()

    def plot_model(self, input_shape, show_shapes=True):
        x = {
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )
