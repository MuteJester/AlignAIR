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
    SinusoidalTokenAndPositionEmbedding, Mish
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


class VDeepJAllignExperimental(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_family_count,
            v_gene_count,
            v_allele_count,
            d_family_count,
            d_gene_count,
            d_allele_count,
            j_gene_count,
            j_allele_count,
            ohe_sub_classes_dict,
            use_gene_masking=False
    ):
        super(VDeepJAllignExperimental, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params

        self.max_seq_length = int(max_seq_length)
        self.v_family_count, self.v_gene_count, self.v_allele_count = (
            v_family_count,
            v_gene_count,
            v_allele_count,
        )
        self.d_family_count, self.d_gene_count, self.d_allele_count = (
            d_family_count,
            d_gene_count,
            d_allele_count,
        )
        self.j_gene_count, self.j_allele_count = j_gene_count, j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.ohe_sub_classes_dict = ohe_sub_classes_dict
        self.transformer_block = TransformerBlock(embed_dim=32, num_heads=8, ff_dim=768)

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_family",
            "v_gene",
            "v_allele",
            "d_family",
            "d_gene",
            "d_allele",
            "j_gene",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.use_gene_masking = use_gene_masking

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):
        self.j_gene_call_middle = Dense(
            self.j_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_gene_call_head = Dense(
            self.j_gene_count, activation="softmax", name="j_gene"
        )  # (v_feature_map)

        self.j_allele_call_middle = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="softmax", name="j_allele"
        )  # (v_feature_map)

        self.j_gene_call_gene_allele_concat = concatenate

    def _init_d_classification_layers(self):
        self.d_family_call_middle = Dense(
            self.d_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_family_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_family_call_head = Dense(
            self.d_family_count, activation="softmax", name="d_family"
        )  # (v_feature_map)

        self.d_gene_call_middle = Dense(
            self.d_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_gene_call_head = Dense(
            self.d_gene_count, activation="softmax", name="d_gene"
        )  # (v_feature_map)

        self.d_allele_call_middle = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="softmax", name="d_allele"
        )  # (v_feature_map)

        self.d_gene_call_family_gene_concat = concatenate
        self.d_gene_call_gene_allele_concat = concatenate

    def _init_v_classification_layers(self):
        self.v_family_call_middle = Dense(
            self.v_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_family_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_family_call_head = Dense(
            self.v_family_count, activation="softmax", name="v_family"
        )  # (v_feature_map)

        self.v_family_dropout = Dropout(0.2)

        self.v_gene_call_middle = Dense(
            self.v_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_gene_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_gene_call_head = Dense(
            self.v_gene_count, activation="softmax", name="v_gene"
        )  # (v_feature_map)
        self.v_gene_dropout = Dropout(0.2)

        self.v_allele_call_middle = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="softmax", name="v_allele"
        )  # (v_feature_map)
        self.v_allele_dropout = Dropout(0.2)
        self.v_allele_feature_distill = Dense(
            self.v_family_count + self.v_gene_count + self.v_allele_count,
            activation=self.classification_middle_layer_activation,
            name="v_gene_allele_distill",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_gene_call_family_gene_concat = concatenate
        self.v_gene_call_gene_allele_concat = concatenate

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_family_middle = self.v_family_call_middle(v_feature_map)
        v_family_middle = self.v_family_dropout(v_family_middle)
        v_family = self.v_family_call_head(v_family_middle)

        if self.use_gene_masking:
            v_family_class = tf.math.argmax(v_family, 1)
            v_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["family"], v_family_class, axis=0
            )

        v_gene_middle = self.v_gene_call_middle(v_feature_map)
        v_gene_middle = self.v_gene_call_family_gene_concat(
            [v_gene_middle, v_family_middle]
        )
        v_gene_middle = self.v_gene_dropout(v_gene_middle)
        v_gene = self.v_gene_call_head(v_gene_middle)

        if self.use_gene_masking:
            v_gene = tf.multiply(v_gene_classes_masks, v_gene)

        # Add advance indexing
        if self.use_gene_masking:
            v_gene_class = tf.math.argmax(v_gene, 1)
            v_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["gene"], v_family_class, axis=0
            )
            v_allele_classes_masks = tf.gather(
                v_allele_classes_masks, v_gene_class, axis=1, batch_dims=1
            )

        v_allele_middle = self.v_allele_call_middle(v_feature_map)
        v_allele_middle = self.v_gene_call_gene_allele_concat(
            [v_family_middle, v_gene_middle, v_allele_middle]
        )
        v_allele_middle = self.v_allele_dropout(v_allele_middle)
        v_allele_middle = self.v_allele_feature_distill(v_allele_middle)
        v_allele = self.v_allele_call_head(v_allele_middle)
        if self.use_gene_masking:
            v_allele = tf.multiply(v_allele_classes_masks, v_allele)
        # ============================ D =============================
        d_family_middle = self.d_family_call_middle(d_feature_map)
        d_family = self.d_family_call_head(d_family_middle)

        if self.use_gene_masking:
            d_family_class = tf.math.argmax(d_family, 1)
            d_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["family"], d_family_class, axis=0
            )

        d_gene_middle = self.d_gene_call_middle(d_feature_map)
        d_gene_middle = self.d_gene_call_family_gene_concat(
            [d_gene_middle, d_family_middle]
        )
        d_gene = self.d_gene_call_head(d_gene_middle)

        if self.use_gene_masking:
            d_gene = tf.multiply(d_gene_classes_masks, d_gene)

        # Add advance indexing
        if self.use_gene_masking:
            d_gene_class = tf.math.argmax(d_gene, 1)
            d_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["gene"], d_family_class, axis=0
            )
            d_allele_classes_masks = tf.gather(
                d_allele_classes_masks, d_gene_class, axis=1, batch_dims=1
            )

        d_allele_middle = self.d_allele_call_middle(d_feature_map)
        d_allele_middle = self.d_gene_call_gene_allele_concat(
            [d_allele_middle, d_gene_middle]
        )
        d_allele = self.d_allele_call_head(d_allele_middle)
        if self.use_gene_masking:
            d_allele = tf.multiply(d_allele_classes_masks, d_allele)
        # ============================ J =============================
        j_gene_middle = self.j_gene_call_middle(j_feature_map)
        j_gene = self.j_gene_call_head(j_gene_middle)

        if self.use_gene_masking:
            j_gene_class = tf.math.argmax(j_gene, 1)
            j_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["J"]["gene"], j_gene_class, axis=0
            )

        j_allele_middle = self.j_allele_call_middle(j_feature_map)
        j_allele_middle = self.j_gene_call_gene_allele_concat(
            [j_allele_middle, j_gene_middle]
        )
        j_allele = self.j_allele_call_head(j_allele_middle)

        if self.use_gene_masking:
            j_allele = tf.multiply(j_allele_classes_masks, j_allele)

        return v_family, v_gene, v_allele, d_family, d_gene, d_allele, j_gene, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)
        concatenated_input_embedding = self.transformer_block(concatenated_input_embedding)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        conv_layer_2 = self.conv_layer_2(conv_layer_1)
        conv_layer_3 = self.conv_layer_3(conv_layer_2)
        last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = last_conv_layer
        concatenated_signals = Flatten()(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            concatenated_signals
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        (
            v_family,
            v_gene,
            v_allele,
            d_family,
            d_gene,
            d_allele,
            j_gene,
            j_allele,
        ) = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_family": v_family,
            "v_gene": v_gene,
            "v_allele": v_allele,
            "d_family": d_family,
            "d_gene": d_gene,
            "d_allele": d_allele,
            "j_gene": j_gene,
            "j_allele": j_allele,
        }

    # def custom_post_processing(self,predictions):
    #     processed_predictions = None

    #     return processed_predictions

    # def predict(self, x,batch_size=None,
    #     verbose="auto",
    #     steps=None,
    #     callbacks=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False):
    #         # Call the predict method of the parent class
    #         predictions = super(VDeepJAllign, self).predict(x,  batch_size=batch_size,
    #                                                             verbose=verbose,
    #                                                             steps=steps,
    #                                                             callbacks=callbacks,
    #                                                             max_queue_size=max_queue_size,
    #                                                             workers=workers,
    #                                                             use_multiprocessing=use_multiprocessing)

    #         # Perform your custom post-processing step on predictions
    #         processed_predictions = self.custom_post_processing(predictions)

    #         return processed_predictions

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def call_hierarchy_loss(
            self, family_true, gene_true, allele_true, family_pred, gene_pred, allele_pred
    ):
        if family_true != None:
            family_loss = K.categorical_crossentropy(
                family_true, family_pred
            )  # K.categorical_crossentropy
        gene_loss = K.categorical_crossentropy(gene_true, gene_pred)
        allele_loss = K.categorical_crossentropy(allele_true, allele_pred)

        # family_loss_mean = K.mean(family_loss)
        # gene_loss_mean = K.mean(gene_loss)
        # allele_loss_mean = K.mean(allele_loss)

        # Penalty for wrong family classification
        penalty_upper = K.constant([10.0])
        penalty_mid = K.constant([5.0])
        penalty_lower = K.constant([1.0])

        if family_true != None:
            family_penalty = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                penalty_upper,
                penalty_lower,
            )
            gene_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_mid,
                penalty_lower,
            )
        else:
            family_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_upper,
                penalty_lower,
            )

        # Compute the final loss based on the constraint
        if family_true != None:
            loss = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                family_penalty * (family_loss + gene_loss + allele_loss),
                K.switch(
                    K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                    family_loss + gene_penalty * (gene_loss + allele_loss),
                    family_loss + gene_loss + penalty_upper * allele_loss,
                ),
            )
        else:
            loss = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                family_penalty * (gene_loss + allele_loss),
                gene_loss + penalty_upper * allele_loss,
            )

        return K.mean(loss)

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mod3_mse_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[0]),
            tf.squeeze(classification_true[1]),
            tf.squeeze(classification_true[2]),
            tf.squeeze(classification_pred[0]),
            tf.squeeze(classification_pred[1]),
            tf.squeeze(classification_pred[2]),
        )

        clf_d_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[3]),
            tf.squeeze(classification_true[4]),
            tf.squeeze(classification_true[5]),
            tf.squeeze(classification_pred[3]),
            tf.squeeze(classification_pred[4]),
            tf.squeeze(classification_pred[5]),
        )

        clf_j_loss = self.call_hierarchy_loss(
            None,
            tf.squeeze(classification_true[6]),
            tf.squeeze(classification_true[7]),
            None,
            tf.squeeze(classification_pred[6]),
            tf.squeeze(classification_pred[7]),
        )

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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

    def _freeze_v_classifier_component(self):
        for layer in [
            self.v_family_call_middle,
            self.v_family_call_head,
            self.v_gene_call_middle,
            self.v_gene_call_head,
            self.v_allele_call_middle,
            self.v_allele_feature_distill,
            self.v_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_d_classifier_component(self):
        for layer in [
            self.d_family_call_middle,
            self.d_family_call_head,
            self.d_gene_call_middle,
            self.d_gene_call_head,
            self.d_allele_call_middle,
            self.d_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_j_classifier_component(self):
        for layer in [
            self.j_gene_call_middle,
            self.j_gene_call_head,
            self.j_allele_call_middle,
            self.j_allele_call_head,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalV2(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_family_count,
            v_gene_count,
            v_allele_count,
            d_family_count,
            d_gene_count,
            d_allele_count,
            j_gene_count,
            j_allele_count,
            ohe_sub_classes_dict,
            use_gene_masking=False
    ):
        super(VDeepJAllignExperimentalV2, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params

        self.max_seq_length = int(max_seq_length)
        self.v_family_count, self.v_gene_count, self.v_allele_count = (
            v_family_count,
            v_gene_count,
            v_allele_count,
        )
        self.d_family_count, self.d_gene_count, self.d_allele_count = (
            d_family_count,
            d_gene_count,
            d_allele_count,
        )
        self.j_gene_count, self.j_allele_count = j_gene_count, j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.ohe_sub_classes_dict = ohe_sub_classes_dict
        self.transformer_blocks = [TransformerBlock(embed_dim=32, num_heads=8, ff_dim=512) for _ in range(5)]

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_family",
            "v_gene",
            "v_allele",
            "d_family",
            "d_gene",
            "d_allele",
            "j_gene",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.use_gene_masking = use_gene_masking

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):
        self.j_gene_call_middle = Dense(
            self.j_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_gene_call_head = Dense(
            self.j_gene_count, activation="softmax", name="j_gene"
        )  # (v_feature_map)

        self.j_allele_call_middle = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="softmax", name="j_allele"
        )  # (v_feature_map)

        self.j_gene_call_gene_allele_concat = concatenate

    def _init_d_classification_layers(self):
        self.d_family_call_middle = Dense(
            self.d_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_family_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_family_call_head = Dense(
            self.d_family_count, activation="softmax", name="d_family"
        )  # (v_feature_map)

        self.d_gene_call_middle = Dense(
            self.d_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_gene_call_head = Dense(
            self.d_gene_count, activation="softmax", name="d_gene"
        )  # (v_feature_map)

        self.d_allele_call_middle = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="softmax", name="d_allele"
        )  # (v_feature_map)

        self.d_gene_call_family_gene_concat = concatenate
        self.d_gene_call_gene_allele_concat = concatenate

    def _init_v_classification_layers(self):
        self.v_family_call_middle = Dense(
            self.v_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_family_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_family_call_head = Dense(
            self.v_family_count, activation="softmax", name="v_family"
        )  # (v_feature_map)

        self.v_family_dropout = Dropout(0.2)

        self.v_gene_call_middle = Dense(
            self.v_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_gene_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_gene_call_head = Dense(
            self.v_gene_count, activation="softmax", name="v_gene"
        )  # (v_feature_map)
        self.v_gene_dropout = Dropout(0.2)

        self.v_allele_call_middle = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="softmax", name="v_allele"
        )  # (v_feature_map)
        self.v_allele_dropout = Dropout(0.2)
        self.v_allele_feature_distill = Dense(
            self.v_family_count + self.v_gene_count + self.v_allele_count,
            activation=self.classification_middle_layer_activation,
            name="v_gene_allele_distill",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_gene_call_family_gene_concat = concatenate
        self.v_gene_call_gene_allele_concat = concatenate

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_family_middle = self.v_family_call_middle(v_feature_map)
        v_family_middle = self.v_family_dropout(v_family_middle)
        v_family = self.v_family_call_head(v_family_middle)

        if self.use_gene_masking:
            v_family_class = tf.math.argmax(v_family, 1)
            v_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["family"], v_family_class, axis=0
            )

        v_gene_middle = self.v_gene_call_middle(v_feature_map)
        v_gene_middle = self.v_gene_call_family_gene_concat(
            [v_gene_middle, v_family_middle]
        )
        v_gene_middle = self.v_gene_dropout(v_gene_middle)
        v_gene = self.v_gene_call_head(v_gene_middle)

        if self.use_gene_masking:
            v_gene = tf.multiply(v_gene_classes_masks, v_gene)

        # Add advance indexing
        if self.use_gene_masking:
            v_gene_class = tf.math.argmax(v_gene, 1)
            v_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["gene"], v_family_class, axis=0
            )
            v_allele_classes_masks = tf.gather(
                v_allele_classes_masks, v_gene_class, axis=1, batch_dims=1
            )

        v_allele_middle = self.v_allele_call_middle(v_feature_map)
        v_allele_middle = self.v_gene_call_gene_allele_concat(
            [v_family_middle, v_gene_middle, v_allele_middle]
        )
        v_allele_middle = self.v_allele_dropout(v_allele_middle)
        v_allele_middle = self.v_allele_feature_distill(v_allele_middle)
        v_allele = self.v_allele_call_head(v_allele_middle)
        if self.use_gene_masking:
            v_allele = tf.multiply(v_allele_classes_masks, v_allele)
        # ============================ D =============================
        d_family_middle = self.d_family_call_middle(d_feature_map)
        d_family = self.d_family_call_head(d_family_middle)

        if self.use_gene_masking:
            d_family_class = tf.math.argmax(d_family, 1)
            d_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["family"], d_family_class, axis=0
            )

        d_gene_middle = self.d_gene_call_middle(d_feature_map)
        d_gene_middle = self.d_gene_call_family_gene_concat(
            [d_gene_middle, d_family_middle]
        )
        d_gene = self.d_gene_call_head(d_gene_middle)

        if self.use_gene_masking:
            d_gene = tf.multiply(d_gene_classes_masks, d_gene)

        # Add advance indexing
        if self.use_gene_masking:
            d_gene_class = tf.math.argmax(d_gene, 1)
            d_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["gene"], d_family_class, axis=0
            )
            d_allele_classes_masks = tf.gather(
                d_allele_classes_masks, d_gene_class, axis=1, batch_dims=1
            )

        d_allele_middle = self.d_allele_call_middle(d_feature_map)
        d_allele_middle = self.d_gene_call_gene_allele_concat(
            [d_allele_middle, d_gene_middle]
        )
        d_allele = self.d_allele_call_head(d_allele_middle)
        if self.use_gene_masking:
            d_allele = tf.multiply(d_allele_classes_masks, d_allele)
        # ============================ J =============================
        j_gene_middle = self.j_gene_call_middle(j_feature_map)
        j_gene = self.j_gene_call_head(j_gene_middle)

        if self.use_gene_masking:
            j_gene_class = tf.math.argmax(j_gene, 1)
            j_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["J"]["gene"], j_gene_class, axis=0
            )

        j_allele_middle = self.j_allele_call_middle(j_feature_map)
        j_allele_middle = self.j_gene_call_gene_allele_concat(
            [j_allele_middle, j_gene_middle]
        )
        j_allele = self.j_allele_call_head(j_allele_middle)

        if self.use_gene_masking:
            j_allele = tf.multiply(j_allele_classes_masks, j_allele)

        return v_family, v_gene, v_allele, d_family, d_gene, d_allele, j_gene, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Flatten or use global average pooling
        x = GlobalAveragePooling1D()(x)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        (
            v_family,
            v_gene,
            v_allele,
            d_family,
            d_gene,
            d_allele,
            j_gene,
            j_allele,
        ) = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_family": v_family,
            "v_gene": v_gene,
            "v_allele": v_allele,
            "d_family": d_family,
            "d_gene": d_gene,
            "d_allele": d_allele,
            "j_gene": j_gene,
            "j_allele": j_allele,
        }

    # def custom_post_processing(self,predictions):
    #     processed_predictions = None

    #     return processed_predictions

    # def predict(self, x,batch_size=None,
    #     verbose="auto",
    #     steps=None,
    #     callbacks=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False):
    #         # Call the predict method of the parent class
    #         predictions = super(VDeepJAllign, self).predict(x,  batch_size=batch_size,
    #                                                             verbose=verbose,
    #                                                             steps=steps,
    #                                                             callbacks=callbacks,
    #                                                             max_queue_size=max_queue_size,
    #                                                             workers=workers,
    #                                                             use_multiprocessing=use_multiprocessing)

    #         # Perform your custom post-processing step on predictions
    #         processed_predictions = self.custom_post_processing(predictions)

    #         return processed_predictions

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def call_hierarchy_loss(
            self, family_true, gene_true, allele_true, family_pred, gene_pred, allele_pred
    ):
        if family_true != None:
            family_loss = K.categorical_crossentropy(
                family_true, family_pred
            )  # K.categorical_crossentropy
        gene_loss = K.categorical_crossentropy(gene_true, gene_pred)
        allele_loss = K.categorical_crossentropy(allele_true, allele_pred)

        # family_loss_mean = K.mean(family_loss)
        # gene_loss_mean = K.mean(gene_loss)
        # allele_loss_mean = K.mean(allele_loss)

        # Penalty for wrong family classification
        penalty_upper = K.constant([10.0])
        penalty_mid = K.constant([5.0])
        penalty_lower = K.constant([1.0])

        if family_true != None:
            family_penalty = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                penalty_upper,
                penalty_lower,
            )
            gene_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_mid,
                penalty_lower,
            )
        else:
            family_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_upper,
                penalty_lower,
            )

        # Compute the final loss based on the constraint
        if family_true != None:
            loss = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                family_penalty * (family_loss + gene_loss + allele_loss),
                K.switch(
                    K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                    family_loss + gene_penalty * (gene_loss + allele_loss),
                    family_loss + gene_loss + penalty_upper * allele_loss,
                ),
            )
        else:
            loss = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                family_penalty * (gene_loss + allele_loss),
                gene_loss + penalty_upper * allele_loss,
            )

        return K.mean(loss)

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mod3_mse_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[0]),
            tf.squeeze(classification_true[1]),
            tf.squeeze(classification_true[2]),
            tf.squeeze(classification_pred[0]),
            tf.squeeze(classification_pred[1]),
            tf.squeeze(classification_pred[2]),
        )

        clf_d_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[3]),
            tf.squeeze(classification_true[4]),
            tf.squeeze(classification_true[5]),
            tf.squeeze(classification_pred[3]),
            tf.squeeze(classification_pred[4]),
            tf.squeeze(classification_pred[5]),
        )

        clf_j_loss = self.call_hierarchy_loss(
            None,
            tf.squeeze(classification_true[6]),
            tf.squeeze(classification_true[7]),
            None,
            tf.squeeze(classification_pred[6]),
            tf.squeeze(classification_pred[7]),
        )

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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

    def _freeze_v_classifier_component(self):
        for layer in [
            self.v_family_call_middle,
            self.v_family_call_head,
            self.v_gene_call_middle,
            self.v_gene_call_head,
            self.v_allele_call_middle,
            self.v_allele_feature_distill,
            self.v_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_d_classifier_component(self):
        for layer in [
            self.d_family_call_middle,
            self.d_family_call_head,
            self.d_gene_call_middle,
            self.d_gene_call_head,
            self.d_allele_call_middle,
            self.d_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_j_classifier_component(self):
        for layer in [
            self.j_gene_call_middle,
            self.j_gene_call_head,
            self.j_allele_call_middle,
            self.j_allele_call_head,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalV3(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_family_count,
            v_gene_count,
            v_allele_count,
            d_family_count,
            d_gene_count,
            d_allele_count,
            j_gene_count,
            j_allele_count,
            ohe_sub_classes_dict,
            use_gene_masking=False
    ):
        super(VDeepJAllignExperimentalV3, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params

        self.max_seq_length = int(max_seq_length)
        self.v_family_count, self.v_gene_count, self.v_allele_count = (
            v_family_count,
            v_gene_count,
            v_allele_count,
        )
        self.d_family_count, self.d_gene_count, self.d_allele_count = (
            d_family_count,
            d_gene_count,
            d_allele_count,
        )
        self.j_gene_count, self.j_allele_count = j_gene_count, j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.ohe_sub_classes_dict = ohe_sub_classes_dict
        self.transformer_blocks = [TransformerBlock(embed_dim=32, num_heads=8, ff_dim=64) for _ in range(6)]

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_family",
            "v_gene",
            "v_allele",
            "d_family",
            "d_gene",
            "d_allele",
            "j_gene",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.use_gene_masking = use_gene_masking

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):
        self.j_gene_call_middle = Dense(
            self.j_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_gene_call_head = Dense(
            self.j_gene_count, activation="softmax", name="j_gene"
        )  # (v_feature_map)

        self.j_allele_call_middle = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="softmax", name="j_allele"
        )  # (v_feature_map)

        self.j_gene_call_gene_allele_concat = concatenate

    def _init_d_classification_layers(self):
        self.d_family_call_middle = Dense(
            self.d_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_family_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_family_call_head = Dense(
            self.d_family_count, activation="softmax", name="d_family"
        )  # (v_feature_map)

        self.d_gene_call_middle = Dense(
            self.d_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_gene_call_head = Dense(
            self.d_gene_count, activation="softmax", name="d_gene"
        )  # (v_feature_map)

        self.d_allele_call_middle = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="softmax", name="d_allele"
        )  # (v_feature_map)

        self.d_gene_call_family_gene_concat = concatenate
        self.d_gene_call_gene_allele_concat = concatenate

    def _init_v_classification_layers(self):
        self.v_family_call_middle = Dense(
            self.v_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_family_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_family_call_head = Dense(
            self.v_family_count, activation="softmax", name="v_family"
        )  # (v_feature_map)

        self.v_family_dropout = Dropout(0.2)

        self.v_gene_call_middle = Dense(
            self.v_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_gene_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_gene_call_head = Dense(
            self.v_gene_count, activation="softmax", name="v_gene"
        )  # (v_feature_map)
        self.v_gene_dropout = Dropout(0.2)

        self.v_allele_call_middle = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="softmax", name="v_allele"
        )  # (v_feature_map)
        self.v_allele_dropout = Dropout(0.2)
        self.v_allele_feature_distill = Dense(
            self.v_family_count + self.v_gene_count + self.v_allele_count,
            activation=self.classification_middle_layer_activation,
            name="v_gene_allele_distill",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_gene_call_family_gene_concat = concatenate
        self.v_gene_call_gene_allele_concat = concatenate

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_family_middle = self.v_family_call_middle(v_feature_map)
        v_family_middle = self.v_family_dropout(v_family_middle)
        v_family = self.v_family_call_head(v_family_middle)

        if self.use_gene_masking:
            v_family_class = tf.math.argmax(v_family, 1)
            v_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["family"], v_family_class, axis=0
            )

        v_gene_middle = self.v_gene_call_middle(v_feature_map)
        v_gene_middle = self.v_gene_call_family_gene_concat(
            [v_gene_middle, v_family_middle]
        )
        v_gene_middle = self.v_gene_dropout(v_gene_middle)
        v_gene = self.v_gene_call_head(v_gene_middle)

        if self.use_gene_masking:
            v_gene = tf.multiply(v_gene_classes_masks, v_gene)

        # Add advance indexing
        if self.use_gene_masking:
            v_gene_class = tf.math.argmax(v_gene, 1)
            v_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["gene"], v_family_class, axis=0
            )
            v_allele_classes_masks = tf.gather(
                v_allele_classes_masks, v_gene_class, axis=1, batch_dims=1
            )

        v_allele_middle = self.v_allele_call_middle(v_feature_map)
        v_allele_middle = self.v_gene_call_gene_allele_concat(
            [v_family_middle, v_gene_middle, v_allele_middle]
        )
        v_allele_middle = self.v_allele_dropout(v_allele_middle)
        v_allele_middle = self.v_allele_feature_distill(v_allele_middle)
        v_allele = self.v_allele_call_head(v_allele_middle)
        if self.use_gene_masking:
            v_allele = tf.multiply(v_allele_classes_masks, v_allele)
        # ============================ D =============================
        d_family_middle = self.d_family_call_middle(d_feature_map)
        d_family = self.d_family_call_head(d_family_middle)

        if self.use_gene_masking:
            d_family_class = tf.math.argmax(d_family, 1)
            d_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["family"], d_family_class, axis=0
            )

        d_gene_middle = self.d_gene_call_middle(d_feature_map)
        d_gene_middle = self.d_gene_call_family_gene_concat(
            [d_gene_middle, d_family_middle]
        )
        d_gene = self.d_gene_call_head(d_gene_middle)

        if self.use_gene_masking:
            d_gene = tf.multiply(d_gene_classes_masks, d_gene)

        # Add advance indexing
        if self.use_gene_masking:
            d_gene_class = tf.math.argmax(d_gene, 1)
            d_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["gene"], d_family_class, axis=0
            )
            d_allele_classes_masks = tf.gather(
                d_allele_classes_masks, d_gene_class, axis=1, batch_dims=1
            )

        d_allele_middle = self.d_allele_call_middle(d_feature_map)
        d_allele_middle = self.d_gene_call_gene_allele_concat(
            [d_allele_middle, d_gene_middle]
        )
        d_allele = self.d_allele_call_head(d_allele_middle)
        if self.use_gene_masking:
            d_allele = tf.multiply(d_allele_classes_masks, d_allele)
        # ============================ J =============================
        j_gene_middle = self.j_gene_call_middle(j_feature_map)
        j_gene = self.j_gene_call_head(j_gene_middle)

        if self.use_gene_masking:
            j_gene_class = tf.math.argmax(j_gene, 1)
            j_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["J"]["gene"], j_gene_class, axis=0
            )

        j_allele_middle = self.j_allele_call_middle(j_feature_map)
        j_allele_middle = self.j_gene_call_gene_allele_concat(
            [j_allele_middle, j_gene_middle]
        )
        j_allele = self.j_allele_call_head(j_allele_middle)

        if self.use_gene_masking:
            j_allele = tf.multiply(j_allele_classes_masks, j_allele)

        return v_family, v_gene, v_allele, d_family, d_gene, d_allele, j_gene, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Flatten or use global average pooling
        x = GlobalAveragePooling1D()(x)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        (
            v_family,
            v_gene,
            v_allele,
            d_family,
            d_gene,
            d_allele,
            j_gene,
            j_allele,
        ) = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_family": v_family,
            "v_gene": v_gene,
            "v_allele": v_allele,
            "d_family": d_family,
            "d_gene": d_gene,
            "d_allele": d_allele,
            "j_gene": j_gene,
            "j_allele": j_allele,
        }

    # def custom_post_processing(self,predictions):
    #     processed_predictions = None

    #     return processed_predictions

    # def predict(self, x,batch_size=None,
    #     verbose="auto",
    #     steps=None,
    #     callbacks=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False):
    #         # Call the predict method of the parent class
    #         predictions = super(VDeepJAllign, self).predict(x,  batch_size=batch_size,
    #                                                             verbose=verbose,
    #                                                             steps=steps,
    #                                                             callbacks=callbacks,
    #                                                             max_queue_size=max_queue_size,
    #                                                             workers=workers,
    #                                                             use_multiprocessing=use_multiprocessing)

    #         # Perform your custom post-processing step on predictions
    #         processed_predictions = self.custom_post_processing(predictions)

    #         return processed_predictions

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def call_hierarchy_loss(
            self, family_true, gene_true, allele_true, family_pred, gene_pred, allele_pred
    ):
        if family_true != None:
            family_loss = K.categorical_crossentropy(
                family_true, family_pred
            )  # K.categorical_crossentropy
        gene_loss = K.categorical_crossentropy(gene_true, gene_pred)
        allele_loss = K.categorical_crossentropy(allele_true, allele_pred)

        # family_loss_mean = K.mean(family_loss)
        # gene_loss_mean = K.mean(gene_loss)
        # allele_loss_mean = K.mean(allele_loss)

        # Penalty for wrong family classification
        penalty_upper = K.constant([10.0])
        penalty_mid = K.constant([5.0])
        penalty_lower = K.constant([1.0])

        if family_true != None:
            family_penalty = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                penalty_upper,
                penalty_lower,
            )
            gene_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_mid,
                penalty_lower,
            )
        else:
            family_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_upper,
                penalty_lower,
            )

        # Compute the final loss based on the constraint
        if family_true != None:
            loss = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                family_penalty * (family_loss + gene_loss + allele_loss),
                K.switch(
                    K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                    family_loss + gene_penalty * (gene_loss + allele_loss),
                    family_loss + gene_loss + penalty_upper * allele_loss,
                ),
            )
        else:
            loss = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                family_penalty * (gene_loss + allele_loss),
                gene_loss + penalty_upper * allele_loss,
            )

        return K.mean(loss)

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mse_no_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[0]),
            tf.squeeze(classification_true[1]),
            tf.squeeze(classification_true[2]),
            tf.squeeze(classification_pred[0]),
            tf.squeeze(classification_pred[1]),
            tf.squeeze(classification_pred[2]),
        )

        clf_d_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[3]),
            tf.squeeze(classification_true[4]),
            tf.squeeze(classification_true[5]),
            tf.squeeze(classification_pred[3]),
            tf.squeeze(classification_pred[4]),
            tf.squeeze(classification_pred[5]),
        )

        clf_j_loss = self.call_hierarchy_loss(
            None,
            tf.squeeze(classification_true[6]),
            tf.squeeze(classification_true[7]),
            None,
            tf.squeeze(classification_pred[6]),
            tf.squeeze(classification_pred[7]),
        )

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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

    def _freeze_v_classifier_component(self):
        for layer in [
            self.v_family_call_middle,
            self.v_family_call_head,
            self.v_gene_call_middle,
            self.v_gene_call_head,
            self.v_allele_call_middle,
            self.v_allele_feature_distill,
            self.v_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_d_classifier_component(self):
        for layer in [
            self.d_family_call_middle,
            self.d_family_call_head,
            self.d_gene_call_middle,
            self.d_gene_call_head,
            self.d_allele_call_middle,
            self.d_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_j_classifier_component(self):
        for layer in [
            self.j_gene_call_middle,
            self.j_gene_call_head,
            self.j_allele_call_middle,
            self.j_allele_call_head,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalV4(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_family_count,
            v_gene_count,
            v_allele_count,
            d_family_count,
            d_gene_count,
            d_allele_count,
            j_gene_count,
            j_allele_count,
            ohe_sub_classes_dict,
            use_gene_masking=False
    ):
        super(VDeepJAllignExperimentalV4, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params

        self.max_seq_length = int(max_seq_length)
        self.v_family_count, self.v_gene_count, self.v_allele_count = (
            v_family_count,
            v_gene_count,
            v_allele_count,
        )
        self.d_family_count, self.d_gene_count, self.d_allele_count = (
            d_family_count,
            d_gene_count,
            d_allele_count,
        )
        self.j_gene_count, self.j_allele_count = j_gene_count, j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.ohe_sub_classes_dict = ohe_sub_classes_dict
        self.transformer_blocks = [TransformerBlock(embed_dim=32, num_heads=8, ff_dim=512) for _ in range(4)]

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_family",
            "v_gene",
            "v_allele",
            "d_family",
            "d_gene",
            "d_allele",
            "j_gene",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.use_gene_masking = use_gene_masking

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):
        self.j_gene_call_middle = Dense(
            self.j_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_gene_call_head = Dense(
            self.j_gene_count, activation="softmax", name="j_gene"
        )  # (v_feature_map)

        self.j_allele_call_middle = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="softmax", name="j_allele"
        )  # (v_feature_map)

        self.j_gene_call_gene_allele_concat = concatenate

    def _init_d_classification_layers(self):
        self.d_family_call_middle = Dense(
            self.d_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_family_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_family_call_head = Dense(
            self.d_family_count, activation="softmax", name="d_family"
        )  # (v_feature_map)

        self.d_gene_call_middle = Dense(
            self.d_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_gene_call_head = Dense(
            self.d_gene_count, activation="softmax", name="d_gene"
        )  # (v_feature_map)

        self.d_allele_call_middle = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="softmax", name="d_allele"
        )  # (v_feature_map)

        self.d_gene_call_family_gene_concat = concatenate
        self.d_gene_call_gene_allele_concat = concatenate

    def _init_v_classification_layers(self):
        self.v_family_call_middle = Dense(
            self.v_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_family_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_family_call_head = Dense(
            self.v_family_count, activation="softmax", name="v_family"
        )  # (v_feature_map)

        self.v_family_dropout = Dropout(0.2)

        self.v_gene_call_middle = Dense(
            self.v_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_gene_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_gene_call_head = Dense(
            self.v_gene_count, activation="softmax", name="v_gene"
        )  # (v_feature_map)
        self.v_gene_dropout = Dropout(0.2)

        self.v_allele_call_middle = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="softmax", name="v_allele"
        )  # (v_feature_map)
        self.v_allele_dropout = Dropout(0.2)
        self.v_allele_feature_distill = Dense(
            self.v_family_count + self.v_gene_count + self.v_allele_count,
            activation=self.classification_middle_layer_activation,
            name="v_gene_allele_distill",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_gene_call_family_gene_concat = concatenate
        self.v_gene_call_gene_allele_concat = concatenate

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_family_middle = self.v_family_call_middle(v_feature_map)
        v_family_middle = self.v_family_dropout(v_family_middle)
        v_family = self.v_family_call_head(v_family_middle)

        if self.use_gene_masking:
            v_family_class = tf.math.argmax(v_family, 1)
            v_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["family"], v_family_class, axis=0
            )

        v_gene_middle = self.v_gene_call_middle(v_feature_map)
        v_gene_middle = self.v_gene_call_family_gene_concat(
            [v_gene_middle, v_family_middle]
        )
        v_gene_middle = self.v_gene_dropout(v_gene_middle)
        v_gene = self.v_gene_call_head(v_gene_middle)

        if self.use_gene_masking:
            v_gene = tf.multiply(v_gene_classes_masks, v_gene)

        # Add advance indexing
        if self.use_gene_masking:
            v_gene_class = tf.math.argmax(v_gene, 1)
            v_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["gene"], v_family_class, axis=0
            )
            v_allele_classes_masks = tf.gather(
                v_allele_classes_masks, v_gene_class, axis=1, batch_dims=1
            )

        v_allele_middle = self.v_allele_call_middle(v_feature_map)
        v_allele_middle = self.v_gene_call_gene_allele_concat(
            [v_family_middle, v_gene_middle, v_allele_middle]
        )
        v_allele_middle = self.v_allele_dropout(v_allele_middle)
        v_allele_middle = self.v_allele_feature_distill(v_allele_middle)
        v_allele = self.v_allele_call_head(v_allele_middle)
        if self.use_gene_masking:
            v_allele = tf.multiply(v_allele_classes_masks, v_allele)
        # ============================ D =============================
        d_family_middle = self.d_family_call_middle(d_feature_map)
        d_family = self.d_family_call_head(d_family_middle)

        if self.use_gene_masking:
            d_family_class = tf.math.argmax(d_family, 1)
            d_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["family"], d_family_class, axis=0
            )

        d_gene_middle = self.d_gene_call_middle(d_feature_map)
        d_gene_middle = self.d_gene_call_family_gene_concat(
            [d_gene_middle, d_family_middle]
        )
        d_gene = self.d_gene_call_head(d_gene_middle)

        if self.use_gene_masking:
            d_gene = tf.multiply(d_gene_classes_masks, d_gene)

        # Add advance indexing
        if self.use_gene_masking:
            d_gene_class = tf.math.argmax(d_gene, 1)
            d_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["gene"], d_family_class, axis=0
            )
            d_allele_classes_masks = tf.gather(
                d_allele_classes_masks, d_gene_class, axis=1, batch_dims=1
            )

        d_allele_middle = self.d_allele_call_middle(d_feature_map)
        d_allele_middle = self.d_gene_call_gene_allele_concat(
            [d_allele_middle, d_gene_middle]
        )
        d_allele = self.d_allele_call_head(d_allele_middle)
        if self.use_gene_masking:
            d_allele = tf.multiply(d_allele_classes_masks, d_allele)
        # ============================ J =============================
        j_gene_middle = self.j_gene_call_middle(j_feature_map)
        j_gene = self.j_gene_call_head(j_gene_middle)

        if self.use_gene_masking:
            j_gene_class = tf.math.argmax(j_gene, 1)
            j_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["J"]["gene"], j_gene_class, axis=0
            )

        j_allele_middle = self.j_allele_call_middle(j_feature_map)
        j_allele_middle = self.j_gene_call_gene_allele_concat(
            [j_allele_middle, j_gene_middle]
        )
        j_allele = self.j_allele_call_head(j_allele_middle)

        if self.use_gene_masking:
            j_allele = tf.multiply(j_allele_classes_masks, j_allele)

        return v_family, v_gene, v_allele, d_family, d_gene, d_allele, j_gene, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Flatten or use global average pooling
        x = GlobalAveragePooling1D()(x)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        (
            v_family,
            v_gene,
            v_allele,
            d_family,
            d_gene,
            d_allele,
            j_gene,
            j_allele,
        ) = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_family": v_family,
            "v_gene": v_gene,
            "v_allele": v_allele,
            "d_family": d_family,
            "d_gene": d_gene,
            "d_allele": d_allele,
            "j_gene": j_gene,
            "j_allele": j_allele,
        }

    # def custom_post_processing(self,predictions):
    #     processed_predictions = None

    #     return processed_predictions

    # def predict(self, x,batch_size=None,
    #     verbose="auto",
    #     steps=None,
    #     callbacks=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False):
    #         # Call the predict method of the parent class
    #         predictions = super(VDeepJAllign, self).predict(x,  batch_size=batch_size,
    #                                                             verbose=verbose,
    #                                                             steps=steps,
    #                                                             callbacks=callbacks,
    #                                                             max_queue_size=max_queue_size,
    #                                                             workers=workers,
    #                                                             use_multiprocessing=use_multiprocessing)

    #         # Perform your custom post-processing step on predictions
    #         processed_predictions = self.custom_post_processing(predictions)

    #         return processed_predictions

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def call_hierarchy_loss(
            self, family_true, gene_true, allele_true, family_pred, gene_pred, allele_pred
    ):
        if family_true != None:
            family_loss = K.categorical_crossentropy(
                family_true, family_pred
            )  # K.categorical_crossentropy
        gene_loss = K.categorical_crossentropy(gene_true, gene_pred)
        allele_loss = K.categorical_crossentropy(allele_true, allele_pred)

        # family_loss_mean = K.mean(family_loss)
        # gene_loss_mean = K.mean(gene_loss)
        # allele_loss_mean = K.mean(allele_loss)

        # Penalty for wrong family classification
        penalty_upper = K.constant([10.0])
        penalty_mid = K.constant([5.0])
        penalty_lower = K.constant([1.0])

        if family_true != None:
            family_penalty = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                penalty_upper,
                penalty_lower,
            )
            gene_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_mid,
                penalty_lower,
            )
        else:
            family_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_upper,
                penalty_lower,
            )

        # Compute the final loss based on the constraint
        if family_true != None:
            loss = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                family_penalty * (family_loss + gene_loss + allele_loss),
                K.switch(
                    K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                    family_loss + gene_penalty * (gene_loss + allele_loss),
                    family_loss + gene_loss + penalty_upper * allele_loss,
                ),
            )
        else:
            loss = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                family_penalty * (gene_loss + allele_loss),
                gene_loss + penalty_upper * allele_loss,
            )

        return K.mean(loss)

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mse_no_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[0]),
            tf.squeeze(classification_true[1]),
            tf.squeeze(classification_true[2]),
            tf.squeeze(classification_pred[0]),
            tf.squeeze(classification_pred[1]),
            tf.squeeze(classification_pred[2]),
        )

        clf_d_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[3]),
            tf.squeeze(classification_true[4]),
            tf.squeeze(classification_true[5]),
            tf.squeeze(classification_pred[3]),
            tf.squeeze(classification_pred[4]),
            tf.squeeze(classification_pred[5]),
        )

        clf_j_loss = self.call_hierarchy_loss(
            None,
            tf.squeeze(classification_true[6]),
            tf.squeeze(classification_true[7]),
            None,
            tf.squeeze(classification_pred[6]),
            tf.squeeze(classification_pred[7]),
        )

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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

    def _freeze_v_classifier_component(self):
        for layer in [
            self.v_family_call_middle,
            self.v_family_call_head,
            self.v_gene_call_middle,
            self.v_gene_call_head,
            self.v_allele_call_middle,
            self.v_allele_feature_distill,
            self.v_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_d_classifier_component(self):
        for layer in [
            self.d_family_call_middle,
            self.d_family_call_head,
            self.d_gene_call_middle,
            self.d_gene_call_head,
            self.d_allele_call_middle,
            self.d_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_j_classifier_component(self):
        for layer in [
            self.j_gene_call_middle,
            self.j_gene_call_head,
            self.j_allele_call_middle,
            self.j_allele_call_head,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalV5(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_family_count,
            v_gene_count,
            v_allele_count,
            d_family_count,
            d_gene_count,
            d_allele_count,
            j_gene_count,
            j_allele_count,
            ohe_sub_classes_dict,
            use_gene_masking=False
    ):
        super(VDeepJAllignExperimentalV5, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params

        self.max_seq_length = int(max_seq_length)
        self.v_family_count, self.v_gene_count, self.v_allele_count = (
            v_family_count,
            v_gene_count,
            v_allele_count,
        )
        self.d_family_count, self.d_gene_count, self.d_allele_count = (
            d_family_count,
            d_gene_count,
            d_allele_count,
        )
        self.j_gene_count, self.j_allele_count = j_gene_count, j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.ohe_sub_classes_dict = ohe_sub_classes_dict
        self.transformer_blocks = [TransformerBlock(embed_dim=32, num_heads=8, ff_dim=256) for _ in range(6)]

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_family",
            "v_gene",
            "v_allele",
            "d_family",
            "d_gene",
            "d_allele",
            "j_gene",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.use_gene_masking = use_gene_masking

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):
        self.j_gene_call_middle = Dense(
            self.j_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_gene_call_head = Dense(
            self.j_gene_count, activation="softmax", name="j_gene"
        )  # (v_feature_map)

        self.j_allele_call_middle = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="softmax", name="j_allele"
        )  # (v_feature_map)

        self.j_gene_call_gene_allele_concat = concatenate

    def _init_d_classification_layers(self):
        self.d_family_call_middle = Dense(
            self.d_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_family_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_family_call_head = Dense(
            self.d_family_count, activation="softmax", name="d_family"
        )  # (v_feature_map)

        self.d_gene_call_middle = Dense(
            self.d_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_gene_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_gene_call_head = Dense(
            self.d_gene_count, activation="softmax", name="d_gene"
        )  # (v_feature_map)

        self.d_allele_call_middle = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.01),
        )
        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="softmax", name="d_allele"
        )  # (v_feature_map)

        self.d_gene_call_family_gene_concat = concatenate
        self.d_gene_call_gene_allele_concat = concatenate

    def _init_v_classification_layers(self):
        self.v_family_call_middle = Dense(
            self.v_family_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_family_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_family_call_head = Dense(
            self.v_family_count, activation="softmax", name="v_family"
        )  # (v_feature_map)

        self.v_family_dropout = Dropout(0.2)

        self.v_gene_call_middle = Dense(
            self.v_gene_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_gene_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_gene_call_head = Dense(
            self.v_gene_count, activation="softmax", name="v_gene"
        )  # (v_feature_map)
        self.v_gene_dropout = Dropout(0.2)

        self.v_allele_call_middle = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )
        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="softmax", name="v_allele"
        )  # (v_feature_map)
        self.v_allele_dropout = Dropout(0.2)
        self.v_allele_feature_distill = Dense(
            self.v_family_count + self.v_gene_count + self.v_allele_count,
            activation=self.classification_middle_layer_activation,
            name="v_gene_allele_distill",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_gene_call_family_gene_concat = concatenate
        self.v_gene_call_gene_allele_concat = concatenate

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_family_middle = self.v_family_call_middle(v_feature_map)
        v_family_middle = self.v_family_dropout(v_family_middle)
        v_family = self.v_family_call_head(v_family_middle)

        if self.use_gene_masking:
            v_family_class = tf.math.argmax(v_family, 1)
            v_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["family"], v_family_class, axis=0
            )

        v_gene_middle = self.v_gene_call_middle(v_feature_map)
        v_gene_middle = self.v_gene_call_family_gene_concat(
            [v_gene_middle, v_family_middle]
        )
        v_gene_middle = self.v_gene_dropout(v_gene_middle)
        v_gene = self.v_gene_call_head(v_gene_middle)

        if self.use_gene_masking:
            v_gene = tf.multiply(v_gene_classes_masks, v_gene)

        # Add advance indexing
        if self.use_gene_masking:
            v_gene_class = tf.math.argmax(v_gene, 1)
            v_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["V"]["gene"], v_family_class, axis=0
            )
            v_allele_classes_masks = tf.gather(
                v_allele_classes_masks, v_gene_class, axis=1, batch_dims=1
            )

        v_allele_middle = self.v_allele_call_middle(v_feature_map)
        v_allele_middle = self.v_gene_call_gene_allele_concat(
            [v_family_middle, v_gene_middle, v_allele_middle]
        )
        v_allele_middle = self.v_allele_dropout(v_allele_middle)
        v_allele_middle = self.v_allele_feature_distill(v_allele_middle)
        v_allele = self.v_allele_call_head(v_allele_middle)
        if self.use_gene_masking:
            v_allele = tf.multiply(v_allele_classes_masks, v_allele)
        # ============================ D =============================
        d_family_middle = self.d_family_call_middle(d_feature_map)
        d_family = self.d_family_call_head(d_family_middle)

        if self.use_gene_masking:
            d_family_class = tf.math.argmax(d_family, 1)
            d_gene_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["family"], d_family_class, axis=0
            )

        d_gene_middle = self.d_gene_call_middle(d_feature_map)
        d_gene_middle = self.d_gene_call_family_gene_concat(
            [d_gene_middle, d_family_middle]
        )
        d_gene = self.d_gene_call_head(d_gene_middle)

        if self.use_gene_masking:
            d_gene = tf.multiply(d_gene_classes_masks, d_gene)

        # Add advance indexing
        if self.use_gene_masking:
            d_gene_class = tf.math.argmax(d_gene, 1)
            d_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["D"]["gene"], d_family_class, axis=0
            )
            d_allele_classes_masks = tf.gather(
                d_allele_classes_masks, d_gene_class, axis=1, batch_dims=1
            )

        d_allele_middle = self.d_allele_call_middle(d_feature_map)
        d_allele_middle = self.d_gene_call_gene_allele_concat(
            [d_allele_middle, d_gene_middle]
        )
        d_allele = self.d_allele_call_head(d_allele_middle)
        if self.use_gene_masking:
            d_allele = tf.multiply(d_allele_classes_masks, d_allele)
        # ============================ J =============================
        j_gene_middle = self.j_gene_call_middle(j_feature_map)
        j_gene = self.j_gene_call_head(j_gene_middle)

        if self.use_gene_masking:
            j_gene_class = tf.math.argmax(j_gene, 1)
            j_allele_classes_masks = tf.gather(
                self.ohe_sub_classes_dict["J"]["gene"], j_gene_class, axis=0
            )

        j_allele_middle = self.j_allele_call_middle(j_feature_map)
        j_allele_middle = self.j_gene_call_gene_allele_concat(
            [j_allele_middle, j_gene_middle]
        )
        j_allele = self.j_allele_call_head(j_allele_middle)

        if self.use_gene_masking:
            j_allele = tf.multiply(j_allele_classes_masks, j_allele)

        return v_family, v_gene, v_allele, d_family, d_gene, d_allele, j_gene, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Flatten or use global average pooling
        x = GlobalAveragePooling1D()(x)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        (
            v_family,
            v_gene,
            v_allele,
            d_family,
            d_gene,
            d_allele,
            j_gene,
            j_allele,
        ) = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_family": v_family,
            "v_gene": v_gene,
            "v_allele": v_allele,
            "d_family": d_family,
            "d_gene": d_gene,
            "d_allele": d_allele,
            "j_gene": j_gene,
            "j_allele": j_allele,
        }

    # def custom_post_processing(self,predictions):
    #     processed_predictions = None

    #     return processed_predictions

    # def predict(self, x,batch_size=None,
    #     verbose="auto",
    #     steps=None,
    #     callbacks=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False):
    #         # Call the predict method of the parent class
    #         predictions = super(VDeepJAllign, self).predict(x,  batch_size=batch_size,
    #                                                             verbose=verbose,
    #                                                             steps=steps,
    #                                                             callbacks=callbacks,
    #                                                             max_queue_size=max_queue_size,
    #                                                             workers=workers,
    #                                                             use_multiprocessing=use_multiprocessing)

    #         # Perform your custom post-processing step on predictions
    #         processed_predictions = self.custom_post_processing(predictions)

    #         return processed_predictions

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def call_hierarchy_loss(
            self, family_true, gene_true, allele_true, family_pred, gene_pred, allele_pred
    ):
        if family_true != None:
            family_loss = K.categorical_crossentropy(
                family_true, family_pred
            )  # K.categorical_crossentropy
        gene_loss = K.categorical_crossentropy(gene_true, gene_pred)
        allele_loss = K.categorical_crossentropy(allele_true, allele_pred)

        # family_loss_mean = K.mean(family_loss)
        # gene_loss_mean = K.mean(gene_loss)
        # allele_loss_mean = K.mean(allele_loss)

        # Penalty for wrong family classification
        penalty_upper = K.constant([10.0])
        penalty_mid = K.constant([5.0])
        penalty_lower = K.constant([1.0])

        if family_true != None:
            family_penalty = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                penalty_upper,
                penalty_lower,
            )
            gene_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_mid,
                penalty_lower,
            )
        else:
            family_penalty = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                penalty_upper,
                penalty_lower,
            )

        # Compute the final loss based on the constraint
        if family_true != None:
            loss = K.switch(
                K.not_equal(K.argmax(family_true), K.argmax(family_pred)),
                family_penalty * (family_loss + gene_loss + allele_loss),
                K.switch(
                    K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                    family_loss + gene_penalty * (gene_loss + allele_loss),
                    family_loss + gene_loss + penalty_upper * allele_loss,
                ),
            )
        else:
            loss = K.switch(
                K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),
                family_penalty * (gene_loss + allele_loss),
                gene_loss + penalty_upper * allele_loss,
            )

        return K.mean(loss)

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        v_start_true, v_end_true, d_start_true, d_end_true, j_start_true, j_end_true = regression_true
        # ========================================================================================================================

        # Compute the intersection loss
        # Compute the intersection for v, d, and j
        v_intersection = K.maximum(0.0, K.minimum(v_end, v_end_true) - K.maximum(v_start, v_start_true))
        d_intersection = K.maximum(0.0, K.minimum(d_end, d_end_true) - K.maximum(d_start, d_start_true))
        j_intersection = K.maximum(0.0, K.minimum(j_end, j_end_true) - K.maximum(j_start, j_start_true))

        # Compute the union for v, d, and j
        v_union = (v_end_true - v_start_true) + (v_end - v_start) - v_intersection
        d_union = (d_end_true - d_start_true) + (d_end - d_start) - d_intersection
        j_union = (j_end_true - j_start_true) + (j_end - j_start) - j_intersection

        # Compute IoU for v, d, and j
        v_iou = v_intersection / (v_union + K.epsilon())
        d_iou = d_intersection / (d_union + K.epsilon())
        j_iou = j_intersection / (j_union + K.epsilon())

        # Compute the total IoU loss (assuming you want the average of the three IoUs)
        _lambda = 100
        total_intersection_loss = 3 * _lambda - _lambda * (v_iou + d_iou + j_iou)

        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mse_no_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[0]),
            tf.squeeze(classification_true[1]),
            tf.squeeze(classification_true[2]),
            tf.squeeze(classification_pred[0]),
            tf.squeeze(classification_pred[1]),
            tf.squeeze(classification_pred[2]),
        )

        clf_d_loss = self.call_hierarchy_loss(
            tf.squeeze(classification_true[3]),
            tf.squeeze(classification_true[4]),
            tf.squeeze(classification_true[5]),
            tf.squeeze(classification_pred[3]),
            tf.squeeze(classification_pred[4]),
            tf.squeeze(classification_pred[5]),
        )

        clf_j_loss = self.call_hierarchy_loss(
            None,
            tf.squeeze(classification_true[6]),
            tf.squeeze(classification_true[7]),
            None,
            tf.squeeze(classification_pred[6]),
            tf.squeeze(classification_pred[7]),
        )

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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

    def _freeze_v_classifier_component(self):
        for layer in [
            self.v_family_call_middle,
            self.v_family_call_head,
            self.v_gene_call_middle,
            self.v_gene_call_head,
            self.v_allele_call_middle,
            self.v_allele_feature_distill,
            self.v_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_d_classifier_component(self):
        for layer in [
            self.d_family_call_middle,
            self.d_family_call_head,
            self.d_gene_call_middle,
            self.d_gene_call_head,
            self.d_allele_call_middle,
            self.d_allele_call_head,
        ]:
            layer.trainable = False

    def _freeze_j_classifier_component(self):
        for layer in [
            self.j_gene_call_middle,
            self.j_gene_call_head,
            self.j_allele_call_middle,
            self.j_allele_call_head,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalSingleBeam(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeam, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.transformer_blocks = [TransformerBlock(embed_dim=32, num_heads=8, ff_dim=64) for _ in range(6)]

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):

        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele"
        )

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele"
        )

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="sigmoid", name="v_allele"
        )

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

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
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Flatten or use global average pooling
        x = GlobalAveragePooling1D()(x)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
        }

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mse_no_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = tf.keras.metrics.binary_crossentropy(classification_true[0], classification_pred[0])
        clf_d_loss = tf.keras.metrics.binary_crossentropy(classification_true[1], classification_pred[1])
        clf_j_loss = tf.keras.metrics.binary_crossentropy(classification_true[2], classification_pred[2])

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalSingleBeam2(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeam2, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.transformer_blocks = [TransformerBlock(embed_dim=32, num_heads=8, ff_dim=64) for _ in range(6)]

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="sigmoid", name="v_allele"
        )

    def _init_j_classification_layers(self):

        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele"
        )

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele"
        )

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_5 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_6 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

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
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = self.conv_v_layer_5(v_feature_map)
        v_feature_map = self.conv_v_layer_6(v_feature_map)

        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Flatten or use global average pooling
        x = GlobalAveragePooling1D()(x)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
        }

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mse_no_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = tf.keras.metrics.binary_crossentropy(classification_true[0], classification_pred[0])
        clf_d_loss = tf.keras.metrics.binary_crossentropy(classification_true[1], classification_pred[1])
        clf_j_loss = tf.keras.metrics.binary_crossentropy(classification_true[2], classification_pred[2])

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalSingleBeamRG(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamRG, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self.transformer_blocks = [TransformerBlock(embed_dim=32, num_heads=8, ff_dim=128) for _ in range(6)]

        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle", kernel_initializer=self.initializer,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
            activity_regularizer=l1(0.01),
        )

        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="sigmoid", name="v_allele"
        )

    def _init_j_classification_layers(self):

        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele"
        )

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele"
        )

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_5 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_6 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
            activity_regularizer=l1(0.01)
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation="relu", name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
            activity_regularizer=l1(0.01)
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation="relu", name="v_end", kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
            activity_regularizer=l1(0.01)
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation="relu", name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
            activity_regularizer=l1(0.01)
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation="relu", name="d_end", kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
            activity_regularizer=l1(0.01)
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation="relu", name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
            activity_regularizer=l1(0.01)
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation="relu", name="j_end", kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

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
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = self.conv_v_layer_5(v_feature_map)
        v_feature_map = self.conv_v_layer_6(v_feature_map)

        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Flatten or use global average pooling
        x = GlobalAveragePooling1D()(x)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
        }

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mse_no_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = tf.keras.metrics.binary_crossentropy(classification_true[0], classification_pred[0])
        clf_d_loss = tf.keras.metrics.binary_crossentropy(classification_true[1], classification_pred[1])
        clf_j_loss = tf.keras.metrics.binary_crossentropy(classification_true[2], classification_pred[2])

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalConvSingleBeam(tf.keras.Model):
    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalConvSingleBeam, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )

        self._init_raw_signals_encoding_layers()
        # Hyperparams + Constants
        self.regression_keys = [
            "v_start",
            "v_end",
            "d_start",
            "d_end",
            "j_start",
            "j_end",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.regression_middle_layer_activation = "swish"

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(
            name="total_classification_loss"
        )

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2, initializer=self.initializer)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2, initializer=self.initializer)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=3, initializer=self.initializer)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_5 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_6 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):

        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele"
        )

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele"
        )

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle",
            kernel_regularizer=regularizers.l2(0.03),
        )

        self.v_allele_call_head = Dense(
            self.v_allele_count, activation="sigmoid", name="v_allele"
        )

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_start_out = Dense(1, activation=self.regression_middle_layer_activation, name="v_start",
                                 kernel_initializer=self.initializer)  # (v_end_mid)

        self.v_end_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation=self.regression_middle_layer_activation, name="v_end",
                               kernel_initializer=self.initializer)  # (v_end_mid)

        self.d_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_start_out = Dense(1, activation=self.regression_middle_layer_activation, name="d_start",
                                 kernel_initializer=self.initializer)  # (d_start_mid)

        self.d_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation=self.regression_middle_layer_activation, name="d_end",
                               kernel_initializer=self.initializer)  # (d_end_mid)

        self.j_start_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_start_out = Dense(1, activation=self.regression_middle_layer_activation, name="j_start",
                                 kernel_initializer=self.initializer)  # (j_start_mid)

        self.j_end_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer
        )  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation=self.regression_middle_layer_activation, name="j_end",
                               kernel_initializer=self.initializer)  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

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
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = self.conv_v_layer_5(v_feature_map)
        v_feature_map = self.conv_v_layer_6(v_feature_map)

        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        x = self.concatenated_input_embedding(input_seq)
        x = self.initial_embedding_attention(
            [x, x]
        )

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = x
        concatenated_signals = Flatten()(concatenated_signals)
        x = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        # conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        # conv_layer_2 = self.conv_layer_2(conv_layer_1)
        # conv_layer_3 = self.conv_layer_3(conv_layer_2)
        # last_conv_layer = self.conv_layer_4(conv_layer_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        # concatenated_signals = last_conv_layer
        # concatenated_signals = Flatten()(concatenated_signals)
        # concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(
            x
        )

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(
            inputs["tokenized_sequence_for_masking"]
        )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor((input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor((input_seq_for_masked, j_mask))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "d_start": d_start,
            "d_end": d_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
        }

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(
            0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)
        ) + K.maximum(0.0, K.minimum(v_end, j_end) - K.maximum(v_start, j_start))
        d_intersection_loss = K.maximum(
            0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)
        ) + K.maximum(0.0, K.minimum(d_end, v_end) - K.maximum(d_start, v_start))
        j_intersection_loss = K.maximum(
            0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end)
        )
        total_intersection_loss = (
                v_intersection_loss + d_intersection_loss + j_intersection_loss
        )
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mse_no_regularization(
            tf.squeeze(K.stack(regression_true)), tf.squeeze(K.stack(regression_pred))
        )
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = tf.keras.metrics.binary_crossentropy(classification_true[0], classification_pred[0])
        clf_d_loss = tf.keras.metrics.binary_crossentropy(classification_true[1], classification_pred[1])
        clf_j_loss = tf.keras.metrics.binary_crossentropy(classification_true[2], classification_pred[2])

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # ========================================================================================================================

        # Combine the two losses using a weighted sum
        total_loss = (
                             (self.regression_weight * mse_loss)
                             + (self.intersection_weight * total_intersection_loss)
                     ) + self.classification_weight * classification_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            (
                loss,
                total_intersection_loss,
                mse_loss,
                classification_loss,
            ) = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["insec_loss"] = self.insec_loss_tracker.result()
        metrics["mod3_mse_loss"] = self.mod3_mse_loss_tracker.result()
        metrics["total_classification_loss"] = self.total_ce_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalSingleBeamConvSegmentation(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentation, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.conv_embedding_attention = Attention()
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")
        self.input_for_masked = Input((self.max_seq_length, 1), name="seq_masked")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_segmentation_1 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=3,
                                                              initializer=self.initializer)

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

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

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_5 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_6 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

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

        return v_segment, d_segment, j_segment

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
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = self.conv_v_layer_5(v_feature_map)
        v_feature_map = self.conv_v_layer_6(v_feature_map)

        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        conv_layer_segmentation_1 = self.conv_layer_segmentation_1(concatenated_input_embedding)
        conv_layer_segmentation_2 = self.conv_layer_segmentation_2(conv_layer_segmentation_1)
        conv_layer_segmentation_3 = self.conv_layer_segmentation_3(conv_layer_segmentation_2)
        last_conv_layer = self.conv_layer_segmentation_4(conv_layer_segmentation_3)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = last_conv_layer
        concatenated_signals = Flatten()(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment = self.predict_segments(concatenated_signals)

        # Get the second copy of the inputs
        # input_seq_for_masked = self.reshape_and_cast_input(
        #     inputs["tokenized_sequence_for_masking"]
        # )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq, v_segment))
        masked_sequence_d = self.d_mask_extractor((input_seq, d_segment))
        masked_sequence_j = self.j_mask_extractor((input_seq, j_segment))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
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
        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss
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
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
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
            "tokenized_sequence_for_masking": Input(shape=input_shape),
            "tokenized_sequence": Input(shape=input_shape),
        }
        return tf.keras.utils.plot_model(
            Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes
        )


class VDeepJAllignExperimentalSingleBeamConvSegmentationV2(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentationV2, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_segmentation_1 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_3 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_5 = Conv1D_and_BatchNorm(filters=128, kernel=5, max_pool=2,
                                                              initializer=self.initializer)

        self.residual_connection_segmentation_conv = Conv1D(64, 5, padding='same',
                                                            kernel_regularizer=regularizers.l2(0.01),
                                                            kernel_initializer=self.initializer)
        self.residual_connection_segmentation_max_pool = MaxPool1D(2)
        self.residual_connection_segmentation_activation = LeakyReLU()
        self.residual_connection_segmentation_batch_norm = BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0,
                                                                              scale=0.02)
        self.residual_connection_segmentation_add = Add()

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

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

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_5 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))
        self.conv_v_layer_6 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                   activation=tf.keras.layers.Activation('tanh'))

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

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

        return v_segment, d_segment, j_segment

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
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = self.conv_v_layer_5(v_feature_map)
        v_feature_map = self.conv_v_layer_6(v_feature_map)

        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

        # Residual
        residual_connection_segmentation_conv = self.residual_connection_segmentation_conv(concatenated_input_embedding)
        residual_connection_segmentation_max_pool = self.residual_connection_segmentation_max_pool(
            residual_connection_segmentation_conv)

        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        conv_layer_segmentation_1 = self.conv_layer_segmentation_1(concatenated_input_embedding)

        conv_layer_segmentation_1_res = self.residual_connection_segmentation_add(
            [conv_layer_segmentation_1, residual_connection_segmentation_max_pool])
        conv_layer_segmentation_1_res = self.residual_connection_segmentation_activation(conv_layer_segmentation_1_res)
        conv_layer_segmentation_1_res = self.residual_connection_segmentation_batch_norm(conv_layer_segmentation_1_res)

        conv_layer_segmentation_2 = self.conv_layer_segmentation_2(conv_layer_segmentation_1_res)
        conv_layer_segmentation_3 = self.conv_layer_segmentation_3(conv_layer_segmentation_2)
        conv_layer_segmentation_4 = self.conv_layer_segmentation_4(conv_layer_segmentation_3)

        last_conv_layer = self.conv_layer_segmentation_5(conv_layer_segmentation_4)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = last_conv_layer
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment = self.predict_segments(concatenated_signals)

        # Get the second copy of the inputs
        # input_seq_for_masked = self.reshape_and_cast_input(
        #     inputs["tokenized_sequence_for_masking"]
        # )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq, v_segment))
        masked_sequence_d = self.d_mask_extractor((input_seq, d_segment))
        masked_sequence_j = self.j_mask_extractor((input_seq, j_segment))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
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
        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss
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
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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


class VDeepJAllignExperimentalSingleBeamConvSegmentationResidual(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentationResidual, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_call_mask = CutoutLayer(
            max_seq_length, "V", name="V_extract"
        )  # (v_end_out)
        self.d_call_mask = CutoutLayer(
            max_seq_length, "D", name="D_extract"
        )  # ([d_start_out,d_end_out])
        self.j_call_mask = CutoutLayer(
            max_seq_length, "J", name="J_extract"
        )  # ([j_start_out,j_end_out])

        self.v_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.d_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))
        self.j_mask_extractor = (
            ExtractGeneMask1D()
        )  # (([input_a_l2,input_t_l2,input_g_l2,input_c_l2],v_call_mask))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

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

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

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

        return v_segment, d_segment, j_segment

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
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment = self.predict_segments(concatenated_signals)

        # Get the second copy of the inputs
        # input_seq_for_masked = self.reshape_and_cast_input(
        #     inputs["tokenized_sequence_for_masking"]
        # )

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor((input_seq, v_segment))
        masked_sequence_d = self.d_mask_extractor((input_seq, d_segment))
        masked_sequence_j = self.j_mask_extractor((input_seq, j_segment))

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(
            masked_sequence_v
        )
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(
            masked_sequence_d
        )
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(
            masked_sequence_j
        )

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
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
        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss
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
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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


class VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRF(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection

    RF:
    Removed second embeddings layer, the first one is used in all locations,
    segmentation mask is applied to embedding vector element wise instead of applying it to the input
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRF, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
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

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
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

    def _init_masked_j_signals_encoding_layers(self):
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
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

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

        return v_segment, d_segment, j_segment

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, concatenated_input_embedding])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, concatenated_input_embedding])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, concatenated_input_embedding])

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
        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss
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
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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


class VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRFSinus(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection

    RF:
    Removed second embeddings layer, the first one is used in all locations,
    segmentation mask is applied to embedding vector element wise instead of applying it to the input
    
    Sinus:
    Replaced the vanila position embeddings in the positional embeddings layer with sinusodial position embeddings
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRFSinus, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = SinusoidalTokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))

        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        self.conv_layer_segmentation_1 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_2 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_3 = Conv1D_and_BatchNorm(filters=64, kernel=4, max_pool=2,
                                                              initializer=self.initializer)
        self.conv_layer_segmentation_4 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
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

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
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

    def _init_masked_j_signals_encoding_layers(self):
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
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

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

        return v_segment, d_segment, j_segment

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, concatenated_input_embedding])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, concatenated_input_embedding])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, concatenated_input_embedding])

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
        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss
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
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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


class VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRF_HP(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection

    RF:
    Removed second embeddings layer, the first one is used in all locations,
    segmentation mask is applied to embedding vector element wise instead of applying it to the input
    
    HP:
    Replace raw signal feature extractor activation function with MISH instead of leaky relu
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentationResidualRF_HP, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
        # Resnet Influenced
        raw_activation = mish
        self.conv_layer_segmentation_1 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2,
                                                              initializer=self.initializer, activation=raw_activation)
        self.conv_layer_segmentation_2 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer, activation=raw_activation)
        self.conv_layer_segmentation_3 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer, activation=raw_activation)
        self.conv_layer_segmentation_4 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2,
                                                              initializer=self.initializer, activation=raw_activation)
        self.conv_layer_segmentation_5 = Conv1D_and_BatchNorm(filters=64, kernel=5, max_pool=2,
                                                              initializer=self.initializer, activation=raw_activation)

        self.residual_connection_segmentation_conv_x_to_1 = Conv1D(64, 5, padding='same',
                                                                   kernel_regularizer=regularizers.l2(0.01),
                                                                   kernel_initializer=self.initializer)
        self.residual_connection_segmentation_max_pool_x_to_1 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_x_to_1 = Mish()
        self.residual_connection_segmentation_add_x_to_1 = Add()

        self.residual_connection_segmentation_max_pool_1_to_3 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_1_to_3 = Mish()
        self.residual_connection_segmentation_add_1_to_3 = Add()

        self.residual_connection_segmentation_max_pool_2_to_4 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_2_to_4 = Mish()
        self.residual_connection_segmentation_add_2_to_4 = Add()

        self.residual_connection_segmentation_max_pool_3_to_5 = MaxPool1D(2)
        self.residual_connection_segmentation_activation_3_to_5 = Mish()
        self.residual_connection_segmentation_add_3_to_5 = Add()

        self.residual_connection_segmentation_max_pool_5_to_d = MaxPool1D(2)
        self.residual_connection_segmentation_activation_5_to_d = Mish()
        self.residual_connection_segmentation_add_5_to_d = Add()

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
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

    def _init_masked_j_signals_encoding_layers(self):
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
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

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

        return v_segment, d_segment, j_segment

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, concatenated_input_embedding])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, concatenated_input_embedding])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, concatenated_input_embedding])

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
        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss
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
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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


class VDeepJAllignExperimentalSingleBeamConvSegmentationResidual_DC_MR(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection

    RF:
    Removed second embeddings layer, the first one is used in all locations,
    segmentation mask is applied to embedding vector element wise instead of applying it to the input

    SegmentationResidual_DC_MR:
    Here we add a label for D allele shorter than 3 nuces "Short-D" and a mutation rate regressor
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentationResidual_DC_MR, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.mutation_rate_loss_tracker = tf.keras.metrics.Mean(
            name="mutation_rate_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
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

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
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

    def _init_masked_j_signals_encoding_layers(self):
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
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

        self.mutation_rate_head = Dense(
            1, activation="sigmoid", name="mutation_rate", kernel_initializer=self.initializer
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

        mutation_rate = self.mutation_rate_head(concatenated_signals)

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, concatenated_input_embedding])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, concatenated_input_embedding])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, concatenated_input_embedding])

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

        mutation_rate_loss = tf.keras.metrics.mean_squared_error(self.c2f32(y_true['mutation_rate']),
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
            self.concatenated_input_embedding,
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


class VDeepJAlignExperimentalSingleBeamConvSegmentationResidualV2(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection

    RF:
    Removed second embeddings layer, the first one is used in all locations,
    segmentation mask is applied to embedding vector element wise instead of applying it to the input

    SegmentationResidual_DC_MR:
    Here we add a label for D allele shorter than 3 nuces "Short-D" and a mutation rate regressor

    VDeepJAllignExperimentalSingleBeamConvSegmentationResidualV2
    Here we added support for both sort D label prediction, mutation rate regressor and deletion probability
    classifiers for each allele.
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
    ):
        super(VDeepJAlignExperimentalSingleBeamConvSegmentationResidualV2, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight, \
        self.mutation_rate_weight, self.deletion_rate = (
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.mutation_rate_loss_tracker = tf.keras.metrics.Mean(
            name="mutation_rate_loss"
        )
        self.deletion_loss_tracker = tf.keras.metrics.Mean(
            name="deletion_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
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

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
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

    def _init_masked_j_signals_encoding_layers(self):
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
        self.v_deletion = Dense(
            1, activation="sigmoid", name="v_deletion"
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

        self.j_deletion = Dense(
            1, activation="sigmoid", name="j_deletion"
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
        self.d_deletion = Dense(
            1, activation="sigmoid", name="d_deletion"
        )

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

        self.mutation_rate_head = Dense(
            1, activation="sigmoid", name="mutation_rate", kernel_initializer=self.initializer
        )

    def predict_segments(self, concatenated_signals):
        v_segment_mid = self.v_segment_mid(concatenated_signals)
        v_segment = self.v_segment_out(v_segment_mid)

        d_segment_mid = self.d_segment_mid(concatenated_signals)
        d_segment = self.d_segment_out(d_segment_mid)

        j_segment_mid = self.j_segment_mid(concatenated_signals)
        j_segment = self.j_segment_out(j_segment_mid)

        mutation_rate = self.mutation_rate_head(concatenated_signals)

        return v_segment, d_segment, j_segment, mutation_rate

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_allele_middle = self.v_allele_mid(v_feature_map)
        v_allele = self.v_allele_call_head(v_allele_middle)
        v_deletions = self.v_deletion(v_feature_map)

        # ============================ D =============================
        d_allele_middle = self.d_allele_mid(d_feature_map)
        d_allele = self.d_allele_call_head(d_allele_middle)
        d_deletions = self.d_deletion(v_feature_map)

        # ============================ J =============================
        j_allele_middle = self.j_allele_mid(j_feature_map)
        j_allele = self.j_allele_call_head(j_allele_middle)
        j_deletions = self.j_deletion(v_feature_map)

        return v_allele, d_allele, j_allele, v_deletions, d_deletions, j_deletions

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, concatenated_input_embedding])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, concatenated_input_embedding])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, concatenated_input_embedding])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(masked_sequence_v)
        d_feature_map = self._encode_masked_d_signal(masked_sequence_d)
        j_feature_map = self._encode_masked_j_signal(masked_sequence_j)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele, v_deletions, d_deletions, j_deletions \
            = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)
        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
            'mutation_rate': mutation_rate,
            'v_deletion': v_deletions,
            'd_deletion': d_deletions,
            'j_deletion': j_deletions
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

        deletion_keys = [g + '_deletion' for g in ['v', 'd', 'j']]
        deletion_true = [self.c2f32(y_true[k]) for k in deletion_keys]
        deletion_pred = [self.c2f32(y_pred[k]) for k in deletion_keys]

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

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # mutation rate loss
        mutation_rate_loss = tf.keras.metrics.mean_squared_error(self.c2f32(y_true['mutation_rate']),
                                                                 self.c2f32(y_pred['mutation_rate']))

        # Compute the deletion loss
        v_deletion_loss = tf.keras.metrics.binary_crossentropy(deletion_true[0], deletion_pred[0])
        d_deletion_loss = tf.keras.metrics.binary_crossentropy(deletion_true[1], deletion_pred[1])
        j_deletion_loss = tf.keras.metrics.binary_crossentropy(deletion_true[2], deletion_pred[2])

        deletion_loss = 0.5*v_deletion_loss + 0.05*d_deletion_loss + 0.45*j_deletion_loss

        # Combine the losses using a weighted sum
        total_loss = (
                self.segmentation_weight * total_segmentation_loss
                + self.intersection_weight * total_intersection_loss
                + self.classification_weight * classification_loss
                + self.mutation_rate_weight * mutation_rate_loss
                + self.deletion_rate * deletion_loss

        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss, \
               deletion_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss, \
                deletion_loss
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
        self.deletion_loss_tracker.update_state(deletion_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()
        metrics["mutation_rate_loss"] = self.mutation_rate_loss_tracker.result()
        metrics["deletion_loss"] = self.deletion_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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

class VDeepJAlignExperimentalSingleBeamConvSegmentationResidualV3(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection

    RF:
    Removed second embeddings layer, the first one is used in all locations,
    segmentation mask is applied to embedding vector element wise instead of applying it to the input

    SegmentationResidual_DC_MR:
    Here we add a label for D allele shorter than 3 nuces "Short-D" and a mutation rate regressor

    VDeepJAllignExperimentalSingleBeamConvSegmentationResidualV2
    Here we added support for both sort D label prediction, mutation rate regressor and deletion probability
    classifiers for each allele.
    """

    """
    VDeepJAlignExperimentalSingleBeamConvSegmentationResidualV3
    remove D deletions as D usually to short and this stops the model from converging 
    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
    ):
        super(VDeepJAlignExperimentalSingleBeamConvSegmentationResidualV2, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight, \
        self.mutation_rate_weight, self.deletion_rate = (
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(
            name="classification_loss"
        )
        self.mutation_rate_loss_tracker = tf.keras.metrics.Mean(
            name="mutation_rate_loss"
        )
        self.deletion_loss_tracker = tf.keras.metrics.Mean(
            name="deletion_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
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

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
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

    def _init_masked_j_signals_encoding_layers(self):
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
        self.v_deletion = Dense(
            1, activation="sigmoid", name="v_deletion"
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

        self.j_deletion = Dense(
            1, activation="sigmoid", name="j_deletion"
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
        self.d_deletion = Dense(
            1, activation="sigmoid", name="d_deletion"
        )

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

        self.mutation_rate_head = Dense(
            1, activation="sigmoid", name="mutation_rate", kernel_initializer=self.initializer
        )

    def predict_segments(self, concatenated_signals):
        v_segment_mid = self.v_segment_mid(concatenated_signals)
        v_segment = self.v_segment_out(v_segment_mid)

        d_segment_mid = self.d_segment_mid(concatenated_signals)
        d_segment = self.d_segment_out(d_segment_mid)

        j_segment_mid = self.j_segment_mid(concatenated_signals)
        j_segment = self.j_segment_out(j_segment_mid)

        mutation_rate = self.mutation_rate_head(concatenated_signals)

        return v_segment, d_segment, j_segment, mutation_rate

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_allele_middle = self.v_allele_mid(v_feature_map)
        v_allele = self.v_allele_call_head(v_allele_middle)
        v_deletions = self.v_deletion(v_feature_map)

        # ============================ D =============================
        d_allele_middle = self.d_allele_mid(d_feature_map)
        d_allele = self.d_allele_call_head(d_allele_middle)
        d_deletions = self.d_deletion(v_feature_map)

        # ============================ J =============================
        j_allele_middle = self.j_allele_mid(j_feature_map)
        j_allele = self.j_allele_call_head(j_allele_middle)
        j_deletions = self.j_deletion(v_feature_map)

        return v_allele, d_allele, j_allele, v_deletions, d_deletions, j_deletions

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, concatenated_input_embedding])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, concatenated_input_embedding])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, concatenated_input_embedding])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(masked_sequence_v)
        d_feature_map = self._encode_masked_d_signal(masked_sequence_d)
        j_feature_map = self._encode_masked_j_signal(masked_sequence_j)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele, v_deletions, d_deletions, j_deletions \
            = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)
        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
            'mutation_rate': mutation_rate,
            'v_deletion': v_deletions,
            'd_deletion': d_deletions,
            'j_deletion': j_deletions
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

        deletion_keys = [g + '_deletion' for g in ['v', 'j']]
        deletion_true = [self.c2f32(y_true[k]) for k in deletion_keys]
        deletion_pred = [self.c2f32(y_pred[k]) for k in deletion_keys]

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

        classification_loss = (
                self.v_class_weight * clf_v_loss
                + self.d_class_weight * clf_d_loss
                + self.j_class_weight * clf_j_loss
        )

        # mutation rate loss
        mutation_rate_loss = tf.keras.metrics.mean_squared_error(self.c2f32(y_true['mutation_rate']),
                                                                 self.c2f32(y_pred['mutation_rate']))

        # Compute the deletion loss
        v_deletion_loss = tf.keras.metrics.binary_crossentropy(deletion_true[0], deletion_pred[0])
        j_deletion_loss = tf.keras.metrics.binary_crossentropy(deletion_true[1], deletion_pred[1])

        deletion_loss = 0.5*v_deletion_loss + 0.5*j_deletion_loss

        # Combine the losses using a weighted sum
        total_loss = (
                self.segmentation_weight * total_segmentation_loss
                + self.intersection_weight * total_intersection_loss
                + self.classification_weight * classification_loss
                + self.mutation_rate_weight * mutation_rate_loss
                + self.deletion_rate * deletion_loss

        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss, \
               deletion_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss, \
                deletion_loss
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
        self.deletion_loss_tracker.update_state(deletion_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()
        metrics["mutation_rate_loss"] = self.mutation_rate_loss_tracker.result()
        metrics["deletion_loss"] = self.deletion_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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


class VDeepJAllignExperimentalSingleBeamConvSegmentationResidualIndel(tf.keras.Model):
    """
    this model replaces the transformer blocks back to Conv Blocks
    and replace mask logic with start and end regression to mask prediction as in actual image segmentation
    tasks
    regularization (L1L2) from segmentation and prediction was removed

    V2:
    expanded some of the layer sizes + residual connection

    RF:
    Removed second embeddings layer, the first one is used in all locations,
    segmentation mask is applied to embedding vector element wise instead of applying it to the input

    ResidualIndel:
    This model include 4 new heads, all 4 are single likelihood classifiers, 3 for each gene deletion presence
    probability and another for the mutation rate
    also this model has a new D label for "short D" to flag when the D allele is too short

    """

    def __init__(
            self,
            max_seq_length,
            v_allele_count,
            d_allele_count,
            j_allele_count,
            V_REF=None
    ):
        super(VDeepJAllignExperimentalSingleBeamConvSegmentationResidualIndel, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)
        # Model Params
        self.V_REF = V_REF
        self.max_seq_length = int(max_seq_length)

        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.mutation_rate_weight, self.deletion_weight, \
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 1, 1, 1, 1, 1

        self.segmentation_weight, self.classification_weight, self.intersection_weight = (
            0.5,
            0.5,
            0.5,
        )
        # Hyperparams + Constants
        self.regression_keys = [
            "v_segment",
            "d_segment",
            "j_segment",
        ]
        self.classification_keys = [
            "v_allele",
            "d_allele",
            "j_allele",
        ]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"

        # Tracking
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        self.classification_loss_tracker = tf.keras.metrics.Mean(name="classification_loss")

        self.mutation_rate_loss_tracker = tf.keras.metrics.Mean(
            name="mutation_rate_loss"
        )
        self.deletions_loss_tracker = tf.keras.metrics.Mean(
            name="deletions_loss"
        )
        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.segmentation_feature_flatten = Flatten()

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        # Init Interval Regression Related Layers
        self._init_segmentation_layers()

        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((512, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((512, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((512, 1))

        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_raw_signals_encoding_layers(self):
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

    def _init_masked_v_signals_encoding_layers(self):
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

    def _init_masked_d_signals_encoding_layers(self):
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

    def _init_masked_j_signals_encoding_layers(self):
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

        self.v_deletion = Dense(
            1, activation="sigmoid", name="v_deletion"
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

        self.j_deletion = Dense(
            1, activation="sigmoid", name="j_deletion"
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

        self.d_deletion = Dense(
            1, activation="sigmoid", name="d_deletion"
        )

    def _init_segmentation_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_segment_mid = Dense(
            128, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.v_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="v_segment",
                                   kernel_initializer=self.initializer)

        self.d_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.d_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="d_segment",
                                   kernel_initializer=self.initializer)  # (d_start_mid)

        self.j_segment_mid = Dense(
            32, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer,
        )  # (concatenated_path)
        self.j_segment_out = Dense(self.max_seq_length, activation="sigmoid", name="j_segment",
                                   kernel_initializer=self.initializer)  # (j_start_mid)

        self.mutation_rate = Dense(
            1, activation="sigmoid", name="mutation_rate"
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

        mutation_rate = self.mutation_rate(concatenated_signals)

        return v_segment, d_segment, j_segment, mutation_rate

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_allele_middle = self.v_allele_mid(v_feature_map)
        v_allele = self.v_allele_call_head(v_allele_middle)
        v_deletion = self.v_deletion(v_feature_map)

        # ============================ D =============================
        d_allele_middle = self.d_allele_mid(d_feature_map)
        d_allele = self.d_allele_call_head(d_allele_middle)
        d_deletion = self.d_deletion(d_feature_map)

        # ============================ J =============================
        j_allele_middle = self.j_allele_mid(j_feature_map)
        j_allele = self.j_allele_call_head(j_allele_middle)
        j_deletion = self.j_deletion(j_feature_map)

        return v_allele, d_allele, j_allele, v_deletion, d_deletion, j_deletion

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)

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

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = conv_layer_segmentation_d_res
        concatenated_signals = self.segmentation_feature_flatten(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)
        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(concatenated_signals)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, concatenated_input_embedding])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, concatenated_input_embedding])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, concatenated_input_embedding])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(masked_sequence_v)
        d_feature_map = self._encode_masked_d_signal(masked_sequence_d)
        j_feature_map = self._encode_masked_j_signal(masked_sequence_j)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele, v_deletion, d_deletion, j_deletion = self._predict_vdj_set(v_feature_map,
                                                                                                 d_feature_map,
                                                                                                 j_feature_map)

        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
            'v_deletion': v_deletion,
            'd_deletion': d_deletion,
            'j_deletion': j_deletion,
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

        mutation_rate_true = self.c2f32(y_true['mutation_rate'])
        mutation_rate_pred = self.c2f32(y_pred['mutation_rate'])

        deletions_true = [self.c2f32(y_true[k]) for k in ['v_deletion', 'd_deletion', 'j_deletion']]
        deletions_pred = [self.c2f32(y_pred[k]) for k in ['v_deletion', 'd_deletion', 'j_deletion']]

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

        mutation_rate_loss = tf.keras.metrics.mean_squared_error(mutation_rate_true, mutation_rate_pred)

        v_deletion_loss = tf.keras.metrics.binary_crossentropy(deletions_true[0], deletions_pred[0])
        d_deletion_loss = tf.keras.metrics.binary_crossentropy(deletions_true[1], deletions_pred[1])
        j_deletion_loss = tf.keras.metrics.binary_crossentropy(deletions_true[2], deletions_pred[2])

        deletion_loss = v_deletion_loss + d_deletion_loss + j_deletion_loss

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
                + self.mutation_rate_weight * mutation_rate_loss
                + self.deletion_weight * deletion_loss
        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, deletion_loss, mutation_rate_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss,
                deletion_loss, mutation_rate_loss
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
        self.deletions_loss_tracker.update_state(deletion_loss)
        self.mutation_rate_loss_tracker.update_state(mutation_rate_loss)

        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()

        metrics["deletion_loss"] = self.deletions_loss_tracker.result()
        metrics["mutation_rate_loss"] = self.mutation_rate_loss_tracker.result()

        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
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
