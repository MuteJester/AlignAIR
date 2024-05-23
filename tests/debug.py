import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from GenAIRR.data import builtin_heavy_chain_data_config
from GenAIRR.utilities import DataConfig
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Multiply,
    Reshape, Flatten
)

from src.AlignAIR.Data import LightChainDataset
from src.AlignAIR.Data.datasetBase import DatasetBase
from src.AlignAIR.Models.HeavyChain.losses import d_loss
from src.AlignAIR.Models.Layers.Layers import ConvResidualFeatureExtractionBlock, Conv1D_and_BatchNorm
from src.AlignAIR.Models.Layers.Layers import TokenAndPositionEmbedding, CutoutLayer, MinMaxValueConstraint


class RegularizedConstrainedLogVar(tf.keras.layers.Layer):
    def __init__(self, initial_value=1.0, min_log_var=-3, max_log_var=1, regularizer_weight=0.01):
        super().__init__()
        self.log_var = self.add_weight(name="log_var",
                                       shape=(),
                                       initializer=tf.keras.initializers.Constant(value=tf.math.log(initial_value)),
                                       constraint=lambda x: tf.clip_by_value(x, min_log_var, max_log_var),
                                       trainable=True)
        self.regularizer_weight = regularizer_weight

    def call(self, inputs):
        regularization_loss = self.regularizer_weight * tf.nn.relu(-self.log_var - 2)  # Soft threshold at log(var)=2
        self.add_loss(regularization_loss)
        return tf.exp(-self.log_var)  # Returns the precision as exp(-log(var))


class HcExperimental_V7(tf.keras.Model):
    """
      The AlignAIRR model for performing segmentation, mutation rate estimation,
      and allele classification tasks in heavy chain sequences.

      Attributes:
          max_seq_length (int): Maximum sequence length.
          v_allele_count (int): Number of V alleles.
          d_allele_count (int): Number of D alleles.
          j_allele_count (int): Number of J alleles.
          ... (other attributes)
      """

    def __init__(self, max_seq_length, v_allele_count, d_allele_count, j_allele_count):
        super(HcExperimental_V7, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.GlorotUniform()  # RandomNormal(mean=0.1, stddev=0.02)

        # Model Params
        self.max_seq_length = int(max_seq_length)
        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count

        # Hyperparams + Constants
        self.classification_keys = ["v_allele", "d_allele", "j_allele"]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (0.5, 0.5, 0.5)

        # Define task-specific log variances for dynamic weighting
        self.log_var_v_start = RegularizedConstrainedLogVar()
        self.log_var_v_end = RegularizedConstrainedLogVar()

        self.log_var_d_start = RegularizedConstrainedLogVar()
        self.log_var_d_end = RegularizedConstrainedLogVar()

        self.log_var_j_start = RegularizedConstrainedLogVar()
        self.log_var_j_end = RegularizedConstrainedLogVar()

        self.log_var_v_classification = RegularizedConstrainedLogVar()
        self.log_var_d_classification = RegularizedConstrainedLogVar()
        self.log_var_j_classification = RegularizedConstrainedLogVar()
        self.log_var_mutation = RegularizedConstrainedLogVar()
        self.log_var_indel = RegularizedConstrainedLogVar()
        self.log_var_productivity = RegularizedConstrainedLogVar()

        # Tracking
        self.setup_performance_metrics()

        self.setup_model()

    def setup_model(self):
        # Init Input Layers
        self._init_input_layers()

        self.input_embeddings = TokenAndPositionEmbedding(
            vocab_size=6, embed_dim=32, maxlen=self.max_seq_length
        )

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self.meta_feature_extractor_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                               num_conv_batch_layers=6,
                                                                               kernel_size=[3, 3, 3, 2, 2, 2, 5],
                                                                               max_pool_size=2,
                                                                               conv_activation=tf.keras.layers.Activation(
                                                                                   'tanh'),
                                                                               initializer=self.initializer)

        self.v_segmentation_feature_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                               num_conv_batch_layers=4,
                                                                               kernel_size=[3, 3, 3, 2, 5],
                                                                               max_pool_size=2,
                                                                               conv_activation=tf.keras.layers.Activation(
                                                                                   'tanh'),
                                                                               initializer=self.initializer)
        self.d_segmentation_feature_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                               num_conv_batch_layers=4,
                                                                               kernel_size=[3, 3, 3, 2, 5],
                                                                               max_pool_size=2,
                                                                               conv_activation=tf.keras.layers.Activation(
                                                                                   'tanh'),
                                                                               initializer=self.initializer)
        self.j_segmentation_feature_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                               num_conv_batch_layers=4,
                                                                               kernel_size=[3, 3, 3, 2, 5],
                                                                               max_pool_size=2,
                                                                               conv_activation=tf.keras.layers.Activation(
                                                                                   'tanh'),
                                                                               initializer=self.initializer)

        self.v_mask_layer = CutoutLayer(gene='V', max_size=self.max_seq_length)
        self.d_mask_layer = CutoutLayer(gene='D', max_size=self.max_seq_length)
        self.j_mask_layer = CutoutLayer(gene='J', max_size=self.max_seq_length)

        # Init V/D/J Masked Input Signal Encoding Layers
        # Init V/D/J Masked Input Signal Encoding Layers
        self.v_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                             num_conv_batch_layers=6,
                                                                             kernel_size=[3, 3, 3, 2, 2, 2, 5],
                                                                             max_pool_size=2,
                                                                             conv_activation=tf.keras.layers.Activation(
                                                                                 'tanh'),
                                                                             initializer=self.initializer)

        self.d_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=64,
                                                                             num_conv_batch_layers=4,
                                                                             kernel_size=[3, 3, 3, 3, 5],
                                                                             max_pool_size=2,
                                                                             conv_activation=tf.keras.layers.Activation(
                                                                                 'tanh'),
                                                                             initializer=self.initializer)

        self.j_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                             num_conv_batch_layers=6,
                                                                             kernel_size=[3, 3, 3, 2, 2, 2, 5],
                                                                             max_pool_size=2,
                                                                             conv_activation=tf.keras.layers.Activation(
                                                                                 'tanh'),
                                                                             initializer=self.initializer)

        # Init Interval Regression Related Layers
        self._init_segmentation_predictions()

        # Init the masking layer that will leverage the predicted segmentation mask
        self._init_masking_layers()

        #  =========== V HEADS ======================
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        self._init_j_classification_layers()

    def setup_performance_metrics(self):
        """
           Sets up metrics to track various aspects of model performance during training.

           This method initializes trackers for different types of losses encountered in the model.
           These include total model loss, intersection loss, segmentation loss, classification loss,
           and mutation rate loss. Each tracker is a Keras Mean metric, which computes the mean of
           the given values, useful for tracking the average loss over time.

           Attributes:
               loss_tracker: Tracks the total model loss.
               intersection_loss_tracker: Tracks the intersection loss, which measures the overlap
                                          between different segments in segmentation tasks.
               total_segmentation_loss_tracker: Tracks the total loss related to segmentation tasks.
               classification_loss_tracker: Tracks the loss associated with the classification tasks.
               mutation_rate_loss_tracker: Tracks the loss associated with the estimation of mutation rates.
           """
        # Track the total model loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # Track the intersection loss
        self.scaled_classification_loss_tracker = tf.keras.metrics.Mean(name="scaled_classification_loss")
        # Track the segmentation loss
        self.scaled_indel_count_loss_tracker = tf.keras.metrics.Mean(name="scaled_indel_count_loss")
        # Track the segmentation loss
        self.scaled_productivity_loss_tracker = tf.keras.metrics.Mean(name="scaled_productivity_loss")
        # Track the classification loss
        self.scaled_mutation_rate_loss_tracker = tf.keras.metrics.Mean(name="scaled_mutation_rate_loss")
        # Track the mutation rate loss
        self.segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_masking_layers(self):
        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((self.max_seq_length, 1))
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((self.max_seq_length, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((self.max_seq_length, 1))

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle", kernel_initializer=self.initializer,
        )

        self.v_allele_call_head = Dense(self.v_allele_count, activation="sigmoid", name="v_allele")

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
        )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele", kernel_regularizer=regularizers.l1(0.01)
        )

    def _init_j_classification_layers(self):
        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
        )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele", kernel_regularizer=regularizers.l1(0.01)
        )

    def _init_segmentation_predictions(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.gelu
        self.v_start_out = Dense(
            1, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer, name='v_start'
        )
        self.v_end_out = Dense(
            1, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer, name='v_end'
        )

        self.d_start_out = Dense(
            1, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer, name='d_start'
        )
        self.d_end_out = Dense(
            1, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer, name='d_end'
        )

        self.j_start_out = Dense(
            1, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer, name='j_start'
        )
        self.j_end_out = Dense(
            1, activation=act, kernel_constraint=unit_norm(), kernel_initializer=self.initializer, name='j_end'
        )

        self.mutation_rate_mid = Dense(
            self.max_seq_length, activation=act, name="mutation_rate_mid", kernel_initializer=self.initializer
        )
        self.mutation_rate_dropout = Dropout(0.05)
        self.mutation_rate_head = Dense(
            1, activation='relu', name="mutation_rate", kernel_initializer=self.initializer
            , kernel_constraint=MinMaxValueConstraint(0, 1)
        )

        self.indel_count_mid = Dense(
            self.max_seq_length, activation=act, name="indel_count_mid", kernel_initializer=self.initializer
        )
        self.indel_count_dropout = Dropout(0.05)
        self.indel_count_head = Dense(
            1, activation='relu', name="indel_count", kernel_initializer=self.initializer
            , kernel_constraint=MinMaxValueConstraint(0, 50)
        )

        self.productivity_feature_block = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=1,
                                                               initializer=self.initializer)
        self.productivity_flatten = Flatten()

        self.productivity_dropout = Dropout(0.05)
        self.productivity_head = Dense(
            1, activation='sigmoid', name="productive", kernel_initializer=self.initializer
        )

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

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        meta_features = self.meta_feature_extractor_block(input_embeddings)
        v_segment_features = self.v_segmentation_feature_block(input_embeddings)
        d_segment_features = self.d_segmentation_feature_block(input_embeddings)
        j_segment_features = self.j_segmentation_feature_block(input_embeddings)

        v_start = self.v_start_out(v_segment_features)
        v_end = self.v_end_out(v_segment_features)

        d_start = self.d_start_out(d_segment_features)
        d_end = self.d_end_out(d_segment_features)

        j_start = self.j_start_out(j_segment_features)
        j_end = self.j_end_out(j_segment_features)

        mutation_rate_mid = self.mutation_rate_mid(meta_features)
        mutation_rate_mid = self.mutation_rate_dropout(mutation_rate_mid)
        mutation_rate = self.mutation_rate_head(mutation_rate_mid)

        indel_count_mid = self.indel_count_mid(meta_features)
        indel_count_dropout = self.indel_count_dropout(indel_count_mid)
        indel_count = self.indel_count_head(indel_count_dropout)

        # productivity_features = self.productivity_feature_block(concatenated_matrix)
        productivity_features = self.productivity_flatten(meta_features)
        productivity_features = self.productivity_dropout(productivity_features)
        is_productive = self.productivity_head(productivity_features)

        reshape_masked_sequence_v = self.v_mask_layer([v_start, v_end])
        reshape_masked_sequence_d = self.d_mask_layer([d_start, d_end])
        reshape_masked_sequence_j = self.j_mask_layer([j_start, j_end])

        reshape_masked_sequence_v = tf.expand_dims(reshape_masked_sequence_v, -1)
        reshape_masked_sequence_d = tf.expand_dims(reshape_masked_sequence_d, -1)
        reshape_masked_sequence_j = tf.expand_dims(reshape_masked_sequence_j, -1)

        # import pdb
        # pdb.set_trace()

        masked_sequence_v = self.v_mask_gate([input_embeddings, reshape_masked_sequence_v])
        masked_sequence_d = self.d_mask_gate([input_embeddings, reshape_masked_sequence_d])
        masked_sequence_j = self.j_mask_gate([input_embeddings, reshape_masked_sequence_j])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self.v_feature_extraction_block(masked_sequence_v)
        d_feature_map = self.d_feature_extraction_block(masked_sequence_d)
        j_feature_map = self.j_feature_extraction_block(masked_sequence_j)

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
            'mutation_rate': mutation_rate,
            'indel_count': indel_count,
            'productive': is_productive
        }

    def get_segmentation_feature_map(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        segmentation_features = self.segmentation_feature_extractor_block(input_embeddings)
        return segmentation_features

    def get_v_latent_dimension(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        v_segment_features = self.v_segmentation_feature_block(input_embeddings)
        v_start = self.v_start_out(v_segment_features)
        v_end = self.v_end_out(v_segment_features)
        reshape_masked_sequence_v = self.v_mask_layer([v_start, v_end])
        reshape_masked_sequence_v = tf.expand_dims(reshape_masked_sequence_v, -1)
        masked_sequence_v = self.v_mask_gate([input_embeddings, reshape_masked_sequence_v])
        v_feature_map = self.v_feature_extraction_block(masked_sequence_v)
        v_allele_latent = self.v_allele_mid(v_feature_map)

        return v_allele_latent

    def get_j_latent_dimension(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)
        j_segment_features = self.j_segmentation_feature_block(input_embeddings)
        j_start = self.j_start_out(j_segment_features)
        j_end = self.j_end_out(j_segment_features)
        reshape_masked_sequence_j = self.j_mask_layer([j_start, j_end])
        reshape_masked_sequence_j = tf.expand_dims(reshape_masked_sequence_j, -1)
        masked_sequence_j = self.j_mask_gate([input_embeddings, reshape_masked_sequence_j])
        j_feature_map = self.j_feature_extraction_block(masked_sequence_j)
        j_allele_latent = self.j_allele_mid(j_feature_map)
        return j_allele_latent

    def get_d_latent_dimension(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)
        d_segment_features = self.d_segmentation_feature_block(input_embeddings)
        d_start = self.d_start_out(d_segment_features)
        d_end = self.d_end_out(d_segment_features)
        reshape_masked_sequence_d = self.d_mask_layer([d_start, d_end])
        reshape_masked_sequence_d = tf.expand_dims(reshape_masked_sequence_d, -1)
        masked_sequence_d = self.d_mask_gate([input_embeddings, reshape_masked_sequence_d])
        d_feature_map = self.d_feature_extraction_block(masked_sequence_d)
        d_allele_latent = self.d_allele_mid(d_feature_map)

        return d_allele_latent

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    @staticmethod
    def compute_iou(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        iou = (intersection + 1e-7) / (sum_ - intersection + 1e-7)
        return K.mean(iou)

    @staticmethod
    def dice_coefficient(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        return (2. * intersection + 1e-7) / (K.sum(y_true) + K.sum(y_pred) + 1e-7)

    @staticmethod
    def constraint_penalty(start_idx, end_idx):
        penalty = tf.maximum(0.0, end_idx - start_idx)
        return penalty ** 2  # Squaring to increase penalty for larger violations

    def hierarchical_loss(self, y_true, y_pred):
        # Extract the segmentation and classification outputs
        # segmentation_true = [self.c2f32(y_true[k]) for k in ['v_segment', 'd_segment', 'j_segment']]
        # segmentation_pred = [self.c2f32(y_pred[k]) for k in ['v_segment', 'd_segment', 'j_segment']]

        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        # Short-D Scaling Factor
        short_d_prob = classification_pred[1][:, -1]  # Probabilities for the bad input
        short_d_scaling_factor = 1.0 - short_d_prob

        # Segmentation Loss
        v_start_loss = tf.keras.losses.mean_absolute_error(y_true['v_start'], y_pred['v_start'])
        v_end_loss = tf.keras.losses.mean_absolute_error(y_true['v_end'], y_pred['v_end'])

        d_start_loss = tf.keras.losses.mean_absolute_error(y_true['d_start'], y_pred['d_start'])
        d_end_loss = tf.keras.losses.mean_absolute_error(y_true['d_end'], y_pred['d_end'])

        j_start_loss = tf.keras.losses.mean_absolute_error(y_true['j_start'], y_pred['j_start'])
        j_end_loss = tf.keras.losses.mean_absolute_error(y_true['j_end'], y_pred['j_end'])

        # v_d_penalty = self.constraint_penalty(y_pred['v_end'], y_pred['d_start'])
        # d_j_penalty = self.constraint_penalty(y_pred['d_end'], y_pred['j_start'])

        # Calculate the precision as the inverse of the exponential of each task's log variance
        weighted_v_start = v_start_loss * self.log_var_v_start(v_start_loss)
        weighted_v_end = v_end_loss * self.log_var_v_end(v_end_loss)

        weighted_d_start = d_start_loss * self.log_var_d_start(d_start_loss) * short_d_scaling_factor
        weighted_d_end = d_end_loss * self.log_var_d_end(d_end_loss) * short_d_scaling_factor

        weighted_j_start = j_start_loss * self.log_var_j_start(j_start_loss)
        weighted_j_end = j_end_loss * self.log_var_j_end(j_end_loss)

        ##########################################################################################################

        segmentation_loss = weighted_v_start + weighted_v_end + weighted_d_start + weighted_d_end + weighted_j_start + weighted_j_end

        # Classification Loss

        clf_v_loss = tf.keras.losses.binary_focal_crossentropy(classification_true[0], classification_pred[0])
        # D loss
        penalty_factor = 1.0
        last_label_penalty_factor = 1.0
        # Binary crossentropy
        bce = tf.keras.metrics.binary_focal_crossentropy(classification_true[1], classification_pred[1])
        # # Calculate the total sum of the prediction vectors
        # total_sum = K.sum(y_pred, axis=-1)
        # # Calculate the threshold which is 90% of the total sum
        # threshold = 0.9 * total_sum
        # # Count how many labels are above this threshold
        # labels_above_threshold = K.sum(K.cast(y_pred > threshold[:, None], tf.float32), axis=1)
        # # Apply penalty if count of labels above threshold is greater than 5
        # extra_penalty = penalty_factor * K.cast(labels_above_threshold > 5, tf.float32)
        # # Additional penalty if the last label's likelihood is above 0.5 and any other label is above zero
        last_label_high_confidence = K.cast(K.greater(classification_pred[1][:, -1], 0.5),
                                            tf.float32)  # Check if last label > 0.5
        other_labels_above_zero = K.cast(K.any(K.greater(classification_pred[1][:, :-1], 0), axis=1),
                                         tf.float32)  # Check if any other label > 0
        last_label_penalty = last_label_penalty_factor * last_label_high_confidence * other_labels_above_zero
        # Combined loss with both penalties
        clf_d_loss = (bce + last_label_penalty)

        clf_j_loss = tf.keras.losses.binary_focal_crossentropy(classification_true[2], classification_pred[2])

        precision_v_classification = self.log_var_v_classification(clf_v_loss)
        precision_d_classification = self.log_var_d_classification(clf_d_loss)
        precision_j_classification = self.log_var_j_classification(clf_j_loss)

        classification_loss = precision_v_classification * clf_v_loss + precision_d_classification * clf_d_loss + precision_j_classification * clf_j_loss

        # Mutation Loss
        mutation_rate_loss = tf.keras.losses.mean_absolute_error(y_true['mutation_rate'], y_pred['mutation_rate'])
        # Indel Count Loss
        indel_count_loss = tf.keras.losses.mean_absolute_error(y_true['indel_count'], y_pred['indel_count'])

        # Compute segmentation confidence directly from segmentation masks
        # Assuming segmentation_true and segmentation_pred are lists of [v_segment, d_segment, j_segment]
        # iou_scores = [self.compute_iou(true, pred) for true, pred in zip(segmentation_true, segmentation_pred)]
        # dice_scores = [self.dice_coefficient(true, pred) for true, pred in zip(segmentation_true, segmentation_pred)]

        # Use either IoU or Dice scores as confidence; here we use IoU
        # segmentation_confidence = K.mean(K.stack(dice_scores), axis=0)

        # Compute Productivity Loss
        productive_loss = tf.keras.losses.binary_crossentropy(y_true['productive'], y_pred['productive'])

        precision_mutation = self.log_var_mutation(mutation_rate_loss)
        precision_indel = self.log_var_indel(indel_count_loss)
        precision_productivity = self.log_var_productivity(productive_loss)

        mutation_rate_loss *= precision_mutation
        indel_count_loss *= precision_indel
        productive_loss *= precision_productivity

        # Get precision (inverse of variance) for each task

        # Sum weighted losses
        total_loss = (segmentation_loss + classification_loss +
                      mutation_rate_loss + indel_count_loss +
                      productive_loss)

        return total_loss, classification_loss, indel_count_loss, mutation_rate_loss, segmentation_loss, productive_loss

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
        # clf_d_loss = tf.keras.metrics.binary_crossentropy(classification_true[1], classification_pred[1])
        clf_d_loss = d_loss(classification_true[1], classification_pred[1], penalty_factor=1,
                            last_label_penalty_factor=3)

        clf_j_loss = tf.keras.metrics.binary_crossentropy(classification_true[2], classification_pred[2])

        mutation_rate_loss = tf.keras.metrics.mean_absolute_error(self.c2f32(y_true['mutation_rate']),
                                                                  self.c2f32(y_pred['mutation_rate']))

        indel_count_loss = tf.keras.metrics.mean_absolute_error(self.c2f32(y_true['indel_count']),
                                                                self.c2f32(y_pred['indel_count']))

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
                + mutation_rate_loss + indel_count_loss

        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss, indel_count_loss

    def train_step(self, data):
        x, y = data

        # import pdb
        # pdb.set_trace()
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # (
            #     total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss,indel_count_loss
            # ) = self.multi_task_loss(y, y_pred)  # loss function

            (
                total_loss, scaled_classification_loss, scaled_indel_count_loss, scaled_mutation_rate_loss,
                segmentation_loss, scaled_productive_loss
            ) = self.hierarchical_loss(y, y_pred)  # loss function

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # import pdb
        # pdb.set_trace()
        self.compiled_metrics.update_state(y, y_pred)
        # for key in y.keys():
        #     #print('====',key)
        #     self.compiled_metrics.update_state(y[key], y_pred[key])

        # self.log_metrics(total_loss, total_intersection_loss, total_segmentation_loss, classification_loss,
        #                  mutation_rate_loss,indel_count_loss)

        self.log_metrics(total_loss, scaled_classification_loss, scaled_indel_count_loss, scaled_mutation_rate_loss,
                         segmentation_loss, scaled_productive_loss)

        metrics = self.get_metrics_log()

        return metrics

    def log_metrics(self, total_loss, scaled_classification_loss, scaled_indel_count_loss, scaled_mutation_rate_loss,
                    segmentation_loss, scaled_productive_loss):
        # Compute our own metrics
        self.loss_tracker.update_state(total_loss)
        self.scaled_classification_loss_tracker.update_state(scaled_classification_loss)
        self.scaled_indel_count_loss_tracker.update_state(scaled_indel_count_loss)
        self.scaled_mutation_rate_loss_tracker.update_state(scaled_mutation_rate_loss)
        self.segmentation_loss_tracker.update_state(segmentation_loss)
        self.scaled_productivity_loss_tracker.update_state(scaled_productive_loss)

    def get_metrics_log(self):
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["segmentation_loss"] = self.segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.scaled_classification_loss_tracker.result()
        metrics["mutation_rate_loss"] = self.scaled_mutation_rate_loss_tracker.result()
        metrics["indel_count_loss"] = self.scaled_indel_count_loss_tracker.result()
        metrics['productive_loss'] = self.scaled_productivity_loss_tracker.result()
        return metrics

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



class HeavyChainDatasetExperimental(DatasetBase):
    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=512, batch_read_file=False,
                 nrows=None, seperator=','):
        super().__init__(data_path, dataconfig, batch_size, max_sequence_length, batch_read_file, nrows, seperator)

        self.required_data_columns = ['sequence', 'v_sequence_start', 'v_sequence_end', 'd_sequence_start',
                                      'd_sequence_end', 'j_sequence_start', 'j_sequence_end', 'v_call',
                                      'd_call', 'j_call', 'mutation_rate', 'indels', 'productive']

    def derive_call_one_hot_representation(self):

        v_alleles = sorted(list(self.v_dict))
        d_alleles = sorted(list(self.d_dict))
        j_alleles = sorted(list(self.j_dict))
        # Add Short D Label as Last Label
        d_alleles = d_alleles + ['Short-D']

        self.v_allele_count = len(v_alleles)
        self.d_allele_count = len(d_alleles)
        self.j_allele_count = len(j_alleles)

        self.v_allele_call_ohe = {f: i for i, f in enumerate(v_alleles)}
        self.d_allele_call_ohe = {f: i for i, f in enumerate(d_alleles)}
        self.j_allele_call_ohe = {f: i for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": self.v_allele_count, "allele_call_ohe": self.v_allele_call_ohe},
            "J": {"allele_count": self.j_allele_count, "allele_call_ohe": self.j_allele_call_ohe},
            "D": {"allele_count": self.d_allele_count, "allele_call_ohe": self.d_allele_call_ohe}
        }

    def derive_call_dictionaries(self):
        self.v_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.v_alleles for j in
                       self.dataconfig.v_alleles[i]}
        self.d_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.d_alleles for j in
                       self.dataconfig.d_alleles[i]}
        self.j_dict = {j.name: j.ungapped_seq.upper() for i in self.dataconfig.j_alleles for j in
                       self.dataconfig.j_alleles[i]}

    def get_ohe_reverse_mapping(self):
        get_reverse_dict = lambda dic: {i: j for j, i in dic.items()}
        call_maps = {
            "v_allele": get_reverse_dict(self.v_allele_call_ohe),
            "d_allele": get_reverse_dict(self.d_allele_call_ohe),
            "j_allele": get_reverse_dict(self.j_allele_call_ohe),
        }
        return call_maps

    def _get_single_batch(self, pointer):
        # Read Batch from Dataset
        batch = self.generate_batch(pointer)
        batch = pd.DataFrame(batch)

        # Encoded sequence in batch and collect the padding sizes applied to each sequences
        encoded_sequences, paddings = self.encode_and_pad_sequences(batch['sequence'])
        # use the padding sizes collected to adjust the start/end positions of the alleles
        for _gene in ['v_sequence', 'd_sequence', 'j_sequence']:
            for _position in ['start', 'end']:
                batch.loc[:, _gene + '_' + _position] += paddings

        x = {"tokenized_sequence": encoded_sequences}

        segments = {'v': [], 'd': [], 'j': []}
        indel_counts = []
        productive = []
        for ax, row in batch.iterrows():
            indels = eval(row['indels'])
            indel_counts.append(len(indels))

        # Convert Comma Seperated Allele Ground Truth Labels into Lists
        v_alleles = batch.v_call.apply(lambda x: set(x.split(',')))
        d_alleles = batch.d_call.apply(lambda x: set(x.split(',')))
        j_alleles = batch.j_call.apply(lambda x: set(x.split(',')))

        y = {
            "v_start": batch.v_sequence_start.values.reshape(-1, 1),
            "v_end": batch.v_sequence_end.values.reshape(-1, 1),
            "d_start": batch.d_sequence_start.values.reshape(-1, 1),
            "d_end": batch.d_sequence_end.values.reshape(-1, 1),
            "j_start": batch.j_sequence_start.values.reshape(-1, 1),
            "j_end": batch.j_sequence_end.values.reshape(-1, 1),
            "v_allele": self.one_hot_encode_allele("V", v_alleles),
            "d_allele": self.one_hot_encode_allele("D", d_alleles),
            "j_allele": self.one_hot_encode_allele("J", j_alleles),
            'mutation_rate': batch.mutation_rate.values.reshape(-1, 1),
            'indel_count': np.array(indel_counts).reshape(-1, 1),
            'productive': np.array([float(eval(i)) for i in batch.productive]).reshape(-1, 1)

        }
        return x, y

    def generate_model_params(self):
        return {
            "max_seq_length": self.max_sequence_length,
            "v_allele_count": self.v_allele_count,
            "d_allele_count": self.d_allele_count,
            "j_allele_count": self.j_allele_count,
        }


from uuid import uuid4
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger


class Trainer:
    def __init__(
            self,
            dataset: DatasetBase,
            model,
            epochs,
            steps_per_epoch,
            num_parallel_calls=8,
            log_to_file=False,
            log_file_name=None,
            log_file_path=None,
            classification_metric='categorical_accuracy',
            regression_metric='mae',
            verbose=0,
            batch_file_reader=False,
            pretrained=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=tf.keras.optimizers.Adam,
            optimizers_params=None,
    ):

        self.pretrained = pretrained
        self.model = model
        self.model_type = model
        self.epochs = epochs
        self.input_size = dataset.max_sequence_length
        self.num_parallel_calls = num_parallel_calls
        self.batch_size = dataset.batch_size
        self.steps_per_epoch = steps_per_epoch
        self.classification_head_metric = classification_metric
        self.segmentation_head_metric = regression_metric
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers
        self.optimizers_params = optimizers_params
        self.history = None
        self.verbose = verbose
        self.log_file_name = log_file_name
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.batch_file_reader = batch_file_reader,
        self.train_dataset = dataset
        self.allele_types = ['v', 'j'] if isinstance(self.train_dataset, LightChainDataset) else ['v', 'd', 'j']

        # Set Up Trainer Instance
        self._load_pretrained_model()  # only if pretrained is not None
        self._compile_model()

    def _load_pretrained_model(self):
        model_params = self.train_dataset.generate_model_params()
        self.model = self.model_type(**model_params)
        if self.pretrained is not None:
            self.model.load_weights(self.pretrained)

    def _compile_model(self):

        metrics = {}
        for gene in self.allele_types:
            for pos in ['start', 'end']:
                metrics[gene + '_' + pos] = self.segmentation_head_metric

        if type(self.classification_head_metric) == list:
            for key, m in zip(self.allele_types, self.classification_head_metric):
                if type(m) == list:
                    for met in m:
                        metrics[key + '_allele'] = met
                else:
                    metrics[key + '_allele'] = m
        else:

            for key in self.allele_types:
                metrics[key + '_allele'] = self.classification_head_metric

        # import pdb
        # pdb.set_trace()
        if self.optimizers_params is not None:
            self.model.compile(optimizer=self.optimizers(**self.optimizers_params),
                               loss=None,
                               metrics=metrics)
        else:
            self.model.compile(optimizer=self.optimizers(),
                               loss=None,
                               metrics=metrics)

    def train(self):

        _callbacks = [] if self.callbacks is None else self.callbacks
        if self.log_to_file:
            if self.log_file_path is None:
                raise ValueError('No log_file_path was given to Trainer')

            file_name = str(uuid4()) + '.csv' if self.log_to_file is None else self.log_file_name
            csv_logger = CSVLogger(self.log_file_path + file_name + '.csv', append=True, separator=',')
            _callbacks.append(csv_logger)

        train_dataset = self.train_dataset.get_train_dataset()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        def preprocess_data(*args):
            return args

        train_dataset = train_dataset.map(
            preprocess_data,
            num_parallel_calls=self.num_parallel_calls
        )

        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch // self.batch_size,
            verbose=self.verbose,
            callbacks=_callbacks

        )

    def save_model(self, path):
        postfix = str(uuid4())
        self.model.save_weights(path + f'{postfix}_weights')
        print(f'Model Saved!\n Location: {path + f"{postfix}_weights"}')
        return path + f'{postfix}_weights'

    def load_model(self, weights_path):
        self.model.load_weights(weights_path)

    def rebuild_model(self):
        self.model = self.model_type(**self.train_dataset.generate_model_params())

    def set_custom_dataset_object(self, dataset_instance):
        self.train_dataset = dataset_instance

    def plot_model(self):
        self.model.plot_model((self.input_size, 1))

    def model_summary(self):
        self.model.model_summary((self.input_size, 1))

    def plot_history(self, write_path=None):
        num_plots = len(self.history.history)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows based on num_plots

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5 * num_rows))

        for i, column in enumerate(list(self.history.history)):
            row = i // num_cols  # Calculate the row index for the current subplot
            col = i % num_cols  # Calculate the column index for the current subplot

            ax = axes[row, col] if num_rows > 1 else axes[col]  # Handle single row case

            ax.plot(self.history.epoch, self.history.history[column])
            ax.set_xlabel('Epoch')
            # ax.set_ylabel(column)
            ax.set_title(column)
            ax.grid(lw=2, ls=':')

        # Remove any empty subplots
        if num_plots < num_rows * num_cols:
            for i in range(num_plots, num_rows * num_cols):
                row = i // num_cols
                col = i % num_cols
                fig.delaxes(axes[row, col])

        plt.tight_layout()
        if write_path is None:
            plt.show()
        else:
            plt.savefig(write_path, dpi=300, facecolor='white')


train_dataset = HeavyChainDatasetExperimental(data_path='./sample_HeavyChain_dataset.csv'
                                              , dataconfig=builtin_heavy_chain_data_config(),
                                              batch_size=32,
                                              max_sequence_length=576,
                                              batch_read_file=True)

trainer = Trainer(
    model=HcExperimental_V7,
    dataset=train_dataset,
    epochs=1,
    steps_per_epoch=1,
    verbose=1,
)
trainer.model.build({'tokenized_sequence': (576, 1)})

MODEL_CHECKPOINT = './AlignAIRR_S5F_OGRDB_Experimental_New_Loss_V7'
trainer.model.load_weights(MODEL_CHECKPOINT)
print(trainer.model.log_var_v_end.weights[0].numpy())