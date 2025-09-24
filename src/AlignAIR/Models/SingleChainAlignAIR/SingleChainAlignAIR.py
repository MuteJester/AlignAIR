import numpy as np
import tensorflow as tf
import keras
from GenAIRR.dataconfig import DataConfig
from keras import Model, regularizers
from keras.constraints import unit_norm
from keras.layers import (
    Dense,
    Input,
    Dropout,
    Multiply,
    Reshape,
    Flatten,
    Activation
)
import os
import json
import logging
from pathlib import Path
from AlignAIR.Serialization.model_bundle import ModelBundleConfig, TrainingMeta, FORMAT_VERSION
from AlignAIR.Serialization.io import save_bundle, load_bundle
from AlignAIR.Serialization.saved_model_wrapper import SavedModelInferenceWrapper
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Assuming these are custom layers from your project structure.
# Ensure the relative paths are correct for your environment.
from ..Layers import Conv1D_and_BatchNorm, CutoutLayer, AverageLastLabel, EntropyMetric, SoftCutoutLayer
from ...Models.Layers import ConvResidualFeatureExtractionBlock, RegularizedConstrainedLogVar
from ...Models.Layers import (
    TokenAndPositionEmbedding, MinMaxValueConstraint
)
# Assuming DataConfig is a custom class you've defined.
# from GenAIRR.dataconfig import DataConfig


class SingleChainAlignAIR(Model):
    """
    The AlignAIRR model for single-chain immunoglobulin sequences.

    This model performs several tasks simultaneously:
    1.  **Segmentation:** Identifies the start and end positions of V, D, and J gene segments.
    2.  **Allele Classification:** Predicts the specific V, D, and J alleles.
    3.  **Mutation Analysis:** Estimates the somatic hypermutation rate and indel count.
    4.  **Productivity Call:** Determines if the sequence is productive.

    The model uses a multi-head architecture with shared and task-specific feature
    extractors. It employs a custom hierarchical loss function with dynamic weighting
    to balance the different tasks during training.

    Attributes:
        max_seq_length (int): Maximum length of input sequences.
        dataconfig (DataConfig): Configuration object containing data-specific parameters
                                 like allele counts.
        v_allele_count (int): Number of V alleles.
        d_allele_count (int): Number of D alleles (if applicable).
        j_allele_count (int): Number of J alleles.
        has_d_gene (bool): Flag indicating if the chain includes a D gene.
    """

    def __init__(self, max_seq_length: int, dataconfig:DataConfig,
                 v_allele_latent_size: Optional[int] = None,
                 d_allele_latent_size: Optional[int] = None,
                 j_allele_latent_size: Optional[int] = None):
        """
        Initializes the SingleChainAlignAIR model.

        Args:
            max_seq_length (int): The maximum sequence length the model can handle.
            dataconfig (DataConfig): An object holding data configuration details,
                                     such as allele counts and metadata.
            v_allele_latent_size (int, optional): Size of the latent dimension for V-allele
                                                  classification. Defaults to None.
            d_allele_latent_size (int, optional): Size of the latent dimension for D-allele
                                                  classification. Defaults to None.
            j_allele_latent_size (int, optional): Size of the latent dimension for J-allele
                                                  classification. Defaults to None.
        """
        super(SingleChainAlignAIR, self).__init__()

        # --- Configuration and Parameters ---
        self.max_seq_length = int(max_seq_length)
        self.dataconfig = dataconfig
        self.has_d_gene = self.dataconfig.metadata.has_d
        self.initializer = keras.initializers.GlorotUniform()

        # Allele counts and latent sizes
        self.v_allele_count = self.dataconfig.number_of_v_alleles
        self.j_allele_count = self.dataconfig.number_of_j_alleles
        self.v_allele_latent_size = v_allele_latent_size
        self.j_allele_latent_size = j_allele_latent_size

        if self.has_d_gene:
            self.d_allele_count = self.dataconfig.number_of_d_alleles
            self.d_allele_latent_size = d_allele_latent_size

        # --- Hyperparameters ---
        self.classification_keys = ["v_allele", "d_allele", "j_allele"] if self.has_d_gene else ["v_allele", "j_allele"]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.fblock_activation = 'tanh'

        # Loss function for classification with label smoothing
        self._bce_loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # --- Model Setup ---
        self.setup_model_layers()
        self.setup_log_variances()
        self.setup_performance_metrics()

    def build(self, input_shape=None):
        """Build the model so it can be summarized/used without an initial call.

        Note: Do NOT call self() here to avoid recursion; just mark as built to
        silence the Keras warning. Variables will be created on first call.
        """
        self.built = True
        try:
            super().build(input_shape)
        except Exception:
            # Some Keras versions may not accept dict input_shape; ignore.
            pass

    def setup_model_layers(self):
        """Initializes all layers used in the model."""
        self._init_input_and_embedding_layers()
        self._init_feature_extractors()
        self._init_segmentation_heads()
        self._init_analysis_heads()
        self._init_classification_heads()
        self._init_masking_layers()

    def setup_log_variances(self):
        """Initializes log variance layers for dynamic loss weighting."""
        self.log_var_v_start = RegularizedConstrainedLogVar(layer_name='log_var_v_start')
        self.log_var_v_end = RegularizedConstrainedLogVar(layer_name='log_var_v_end')
        self.log_var_j_start = RegularizedConstrainedLogVar(layer_name='log_var_j_start')
        self.log_var_j_end = RegularizedConstrainedLogVar(layer_name='log_var_j_end')
        self.log_var_v_classification = RegularizedConstrainedLogVar(layer_name='log_var_v_clf')
        self.log_var_j_classification = RegularizedConstrainedLogVar(layer_name='log_var_j_clf')
        self.log_var_mutation = RegularizedConstrainedLogVar(layer_name='log_var_mutation')
        self.log_var_indel = RegularizedConstrainedLogVar(layer_name='log_var_indel')
        self.log_var_productivity = RegularizedConstrainedLogVar(layer_name='log_var_productivity')

        if self.has_d_gene:
            self.log_var_d_start = RegularizedConstrainedLogVar(layer_name='log_var_d_start')
            self.log_var_d_end = RegularizedConstrainedLogVar(layer_name='log_var_d_end')
            self.log_var_d_classification = RegularizedConstrainedLogVar(layer_name='log_var_d_clf')

    def setup_performance_metrics(self):
        """Initializes metrics to track model performance during training."""
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.scaled_classification_loss_tracker = keras.metrics.Mean(name="scaled_classification_loss")
        self.scaled_indel_count_loss_tracker = keras.metrics.Mean(name="scaled_indel_count_loss")
        self.scaled_productivity_loss_tracker = keras.metrics.Mean(name="scaled_productivity_loss")
        self.scaled_mutation_rate_loss_tracker = keras.metrics.Mean(name="scaled_mutation_rate_loss")
        self.segmentation_loss_tracker = keras.metrics.Mean(name="segmentation_loss")

        # Custom metrics for deeper monitoring
        self.average_last_label_tracker = AverageLastLabel(name="average_last_label")
        self.v_allele_entropy_tracker = EntropyMetric(allele_name="v_allele")
        self.j_allele_entropy_tracker = EntropyMetric(allele_name="j_allele")

        # AUC metrics for allele classification heads (epoch-aggregated)
        self.v_allele_auc = keras.metrics.AUC(name="v_allele_auc", multi_label=True)
        self.j_allele_auc = keras.metrics.AUC(name="j_allele_auc", multi_label=True)

        if self.has_d_gene:
            self.d_allele_entropy_tracker = EntropyMetric(allele_name="d_allele")
            self.d_allele_auc = keras.metrics.AUC(name="d_allele_auc", multi_label=True)

        # Boundary accuracy and error metrics (exact and within 1 nt)
        # V
        self.v_start_mae = keras.metrics.Mean(name="v_start_mae")
        self.v_end_mae = keras.metrics.Mean(name="v_end_mae")
        self.v_start_acc = keras.metrics.Mean(name="v_start_acc")  # exact match
        self.v_end_acc = keras.metrics.Mean(name="v_end_acc")
        self.v_start_acc_1nt = keras.metrics.Mean(name="v_start_acc_1nt")  # |err|<=1
        self.v_end_acc_1nt = keras.metrics.Mean(name="v_end_acc_1nt")
        # J
        self.j_start_mae = keras.metrics.Mean(name="j_start_mae")
        self.j_end_mae = keras.metrics.Mean(name="j_end_mae")
        self.j_start_acc = keras.metrics.Mean(name="j_start_acc")
        self.j_end_acc = keras.metrics.Mean(name="j_end_acc")
        self.j_start_acc_1nt = keras.metrics.Mean(name="j_start_acc_1nt")
        self.j_end_acc_1nt = keras.metrics.Mean(name="j_end_acc_1nt")
        # D (optional)
        if self.has_d_gene:
            self.d_start_mae = keras.metrics.Mean(name="d_start_mae")
            self.d_end_mae = keras.metrics.Mean(name="d_end_mae")
            self.d_start_acc = keras.metrics.Mean(name="d_start_acc")
            self.d_end_acc = keras.metrics.Mean(name="d_end_acc")
            self.d_start_acc_1nt = keras.metrics.Mean(name="d_start_acc_1nt")
            self.d_end_acc_1nt = keras.metrics.Mean(name="d_end_acc_1nt")

    def _init_input_and_embedding_layers(self):
        """Initializes input and embedding layers."""
        self.input_layer = Input((self.max_seq_length, 1), name="seq_init")
        self.input_embeddings = TokenAndPositionEmbedding(
            vocab_size=6, embed_dim=32, maxlen=self.max_seq_length, name="tokpos_emb"
        )

    def _init_feature_extractors(self):
        """Initializes convolutional feature extraction blocks."""
        conv_activation = Activation(self.fblock_activation)

        # Shared block for meta features (mutation, indel, productivity)
        self.meta_feature_extractor_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer, name="meta_fblock"
        )

        # Blocks for segmentation feature extraction
        self.v_segmentation_feature_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer, name="v_seg_fblock"
        )
        self.j_segmentation_feature_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer, name="j_seg_fblock"
        )

        # Blocks for classification feature extraction (on masked sequences)
        self.v_feature_extraction_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=6, kernel_size=[3, 3, 3, 2, 2, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer, name="v_cls_fblock"
        )
        self.j_feature_extraction_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=6, kernel_size=[3, 3, 3, 2, 2, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer, name="j_cls_fblock"
        )

        if self.has_d_gene:
            self.d_segmentation_feature_block = ConvResidualFeatureExtractionBlock(
                filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
                max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer, name="d_seg_fblock"
            )
            self.d_feature_extraction_block = ConvResidualFeatureExtractionBlock(
                filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 2, 2, 5],
                max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer, name="d_cls_fblock"
            )

    def _init_segmentation_heads(self):
        """Initializes the output layers for segment boundaries."""
        # Position-logit heads (Dense(L) each). Softmax will be applied in call().
        units = self.max_seq_length
        self.v_start_head = Dense(units, activation=None, name='v_start_logits')
        self.v_end_head = Dense(units, activation=None, name='v_end_logits')
        self.j_start_head = Dense(units, activation=None, name='j_start_logits')
        self.j_end_head = Dense(units, activation=None, name='j_end_logits')

        if self.has_d_gene:
            self.d_start_head = Dense(units, activation=None, name='d_start_logits')
            self.d_end_head = Dense(units, activation=None, name='d_end_logits')

    def _init_analysis_heads(self):
        """Initializes heads for mutation rate, indel count, and productivity."""
        act = keras.activations.gelu

        # Mutation Rate Head
        self.mutation_rate_mid = Dense(self.max_seq_length, activation=act, name="mutation_rate_mid")
        self.mutation_rate_dropout = Dropout(0.05)
        self.mutation_rate_head = Dense(1, activation='relu', name="mutation_rate", kernel_constraint=MinMaxValueConstraint(0, 1))

        # Indel Count Head
        self.indel_count_mid = Dense(self.max_seq_length, activation=act, name="indel_count_mid")
        self.indel_count_dropout = Dropout(0.05)
        self.indel_count_head = Dense(1, activation='relu', name="indel_count", kernel_constraint=MinMaxValueConstraint(0, 50))

        # Productivity Head
        self.productivity_flatten = Flatten()
        self.productivity_dropout = Dropout(0.05)
        self.productivity_head = Dense(1, activation='sigmoid', name="productive")

    def _init_classification_heads(self):
        """Initializes the output layers for allele classification."""
        # V-allele classification
        v_latent_dim = self.v_allele_latent_size or self.v_allele_count * self.latent_size_factor
        self.v_allele_mid = Dense(v_latent_dim, activation=self.classification_middle_layer_activation, name="v_allele_middle")
        self.v_allele_call_head = Dense(self.v_allele_count, activation="sigmoid", name="v_allele")

        # J-allele classification
        j_latent_dim = self.j_allele_latent_size or self.j_allele_count * self.latent_size_factor
        self.j_allele_mid = Dense(j_latent_dim, activation=self.classification_middle_layer_activation, name="j_allele_middle")
        self.j_allele_call_head = Dense(self.j_allele_count, activation="sigmoid", name="j_allele", kernel_regularizer=regularizers.l1(0.01))

        # D-allele classification (if applicable)
        if self.has_d_gene:
            d_latent_dim = self.d_allele_latent_size or self.d_allele_count * self.latent_size_factor
            self.d_allele_mid = Dense(d_latent_dim, activation=self.classification_middle_layer_activation, name="d_allele_middle", kernel_regularizer=regularizers.l2(0.01))
            self.d_allele_call_head = Dense(self.d_allele_count, activation="sigmoid", name="d_allele", kernel_regularizer=regularizers.l2(0.05))

    def _init_masking_layers(self):
        """Initializes layers for creating and applying segmentation masks."""
        # Use soft cutout to preserve gradients near boundaries and respect [start:end)
        self.v_mask_layer = SoftCutoutLayer(gene='V', max_size=self.max_seq_length, k=3.0, name="v_soft_mask")
        self.j_mask_layer = SoftCutoutLayer(gene='J', max_size=self.max_seq_length, k=3.0, name="j_soft_mask")

        self.v_mask_gate = Multiply(name="v_mask_gate")
        self.j_mask_gate = Multiply(name="j_mask_gate")

        self.v_mask_reshape = Reshape((self.max_seq_length, 1), name="v_mask_reshape")
        self.j_mask_reshape = Reshape((self.max_seq_length, 1), name="j_mask_reshape")

        if self.has_d_gene:
            self.d_mask_layer = SoftCutoutLayer(gene='D', max_size=self.max_seq_length, k=3.0, name="d_soft_mask")
            self.d_mask_gate = Multiply(name="d_mask_gate")
            self.d_mask_reshape = Reshape((self.max_seq_length, 1), name="d_mask_reshape")

    def call(self, inputs, training=False):
        """
        Performs the forward pass of the model.

        Args:
            inputs (dict): A dictionary containing the input tensor under the key
                           'tokenized_sequence'.
            training (bool): Flag indicating if the model is in training mode.

        Returns:
            dict: A dictionary of output tensors for each model head.
        """
        input_seq = tf.reshape(inputs["tokenized_sequence"], (-1, self.max_seq_length))
        input_seq = tf.cast(input_seq, "float32")
        input_embeddings = self.input_embeddings(input_seq)
        # 1. Feature Extraction for different tasks
        meta_features = self.meta_feature_extractor_block(input_embeddings)
        v_segment_features = self.v_segmentation_feature_block(input_embeddings)
        j_segment_features = self.j_segmentation_feature_block(input_embeddings)

        # 2. Predict per-position logits for boundaries and turn into probabilities
        v_start_logits = self.v_start_head(v_segment_features)
        v_end_logits = self.v_end_head(v_segment_features)
        j_start_logits = self.j_start_head(j_segment_features)
        j_end_logits = self.j_end_head(j_segment_features)

        v_start_probs = tf.nn.softmax(v_start_logits, axis=-1)
        v_end_probs = tf.nn.softmax(v_end_logits, axis=-1)
        j_start_probs = tf.nn.softmax(j_start_logits, axis=-1)
        j_end_probs = tf.nn.softmax(j_end_logits, axis=-1)

        # Expectations s̄, ē for soft, differentiable gating
        positions = tf.cast(tf.range(self.max_seq_length), tf.float32)[tf.newaxis, :]  # (1, L)
        v_start_exp = tf.reduce_sum(v_start_probs * positions, axis=-1, keepdims=True)
        v_end_exp = tf.reduce_sum(v_end_probs * positions, axis=-1, keepdims=True)
        j_start_exp = tf.reduce_sum(j_start_probs * positions, axis=-1, keepdims=True)
        j_end_exp = tf.reduce_sum(j_end_probs * positions, axis=-1, keepdims=True)

        # 3. Predict Mutation Rate, Indels, and Productivity
        mutation_rate_mid = self.mutation_rate_mid(meta_features)
        mutation_rate_mid = self.mutation_rate_dropout(mutation_rate_mid, training=training)
        mutation_rate = self.mutation_rate_head(mutation_rate_mid)

        indel_count_mid = self.indel_count_mid(meta_features)
        indel_count_mid = self.indel_count_dropout(indel_count_mid, training=training)
        indel_count = self.indel_count_head(indel_count_mid)

        productivity_features = self.productivity_flatten(meta_features)
        productivity_features = self.productivity_dropout(productivity_features, training=training)
        is_productive = self.productivity_head(productivity_features)

        # 4. Create and Apply Masks based on predicted boundaries
        v_mask = self.v_mask_layer([v_start_exp, v_end_exp])
        v_mask = self.v_mask_reshape(v_mask)
        masked_sequence_v = self.v_mask_gate([input_embeddings, v_mask])

        j_mask = self.j_mask_layer([j_start_exp, j_end_exp])
        j_mask = self.j_mask_reshape(j_mask)
        masked_sequence_j = self.j_mask_gate([input_embeddings, j_mask])

        # 5. Extract features from masked sequences for classification
        v_feature_map = self.v_feature_extraction_block(masked_sequence_v)
        j_feature_map = self.j_feature_extraction_block(masked_sequence_j)

        # 6. Predict Alleles
        v_allele_latent = self.v_allele_mid(v_feature_map)
        v_allele = self.v_allele_call_head(v_allele_latent)

        j_allele_latent = self.j_allele_mid(j_feature_map)
        j_allele = self.j_allele_call_head(j_allele_latent)

        # --- D-Gene Specific Path ---
        if self.has_d_gene:
            d_segment_features = self.d_segmentation_feature_block(input_embeddings)
            d_start_logits = self.d_start_head(d_segment_features)
            d_end_logits = self.d_end_head(d_segment_features)

            d_start_probs = tf.nn.softmax(d_start_logits, axis=-1)
            d_end_probs = tf.nn.softmax(d_end_logits, axis=-1)
            d_start_exp = tf.reduce_sum(d_start_probs * positions, axis=-1, keepdims=True)
            d_end_exp = tf.reduce_sum(d_end_probs * positions, axis=-1, keepdims=True)

            d_mask = self.d_mask_layer([d_start_exp, d_end_exp])
            d_mask = self.d_mask_reshape(d_mask)
            masked_sequence_d = self.d_mask_gate([input_embeddings, d_mask])

            d_feature_map = self.d_feature_extraction_block(masked_sequence_d)
            d_allele_latent = self.d_allele_mid(d_feature_map)
            d_allele = self.d_allele_call_head(d_allele_latent)

        # 7. Compile final output dictionary
        # Expose logits for training with CE; also expose expectations for backward-compat.
        output = {
            "v_start_logits": v_start_logits, "v_end_logits": v_end_logits,
            "j_start_logits": j_start_logits, "j_end_logits": j_end_logits,
            # expectations (floats) for compatibility with downstream consumers
            "v_start": v_start_exp, "v_end": v_end_exp,
            "j_start": j_start_exp, "j_end": j_end_exp,
            "v_allele": v_allele, "j_allele": j_allele,
            'mutation_rate': mutation_rate,
            'indel_count': indel_count,
            'productive': is_productive
        }
        if self.has_d_gene:
            output.update({


                'd_start_logits': d_start_logits, 'd_end_logits': d_end_logits,
                'd_start': d_start_exp, 'd_end': d_end_exp, 'd_allele': d_allele
            })

        return output

    def hierarchical_loss(self, y_true, y_pred):
        """
        Calculates the total loss as a dynamically weighted sum of task-specific losses.
        """
        # Helper: create soft target distributions around GT index
        def soft_targets(gt, L, sigma=1.5):
            gt = tf.cast(tf.round(gt), tf.float32)
            gt = tf.clip_by_value(gt, 0.0, float(L - 1))
            positions = tf.cast(tf.range(L), tf.float32)[tf.newaxis, :]
            dist2 = tf.square(positions - gt)
            logits = -0.5 * dist2 / (sigma * sigma)
            probs = tf.nn.softmax(logits, axis=-1)
            return probs

        # Helper: expectation from logits
        def expectation_from_logits(logits):
            probs = tf.nn.softmax(logits, axis=-1)
            pos = tf.cast(tf.range(self.max_seq_length), tf.float32)[tf.newaxis, :]
            return tf.reduce_sum(probs * pos, axis=-1, keepdims=True)

        # --- Segmentation Loss (soft-label CE) ---
        L = self.max_seq_length

        # V boundaries
        v_start_t = soft_targets(y_true['v_start'], L)
        v_end_t = soft_targets(y_true['v_end'], L)
        v_start_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=v_start_t, logits=y_pred['v_start_logits'])
        )
        v_end_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=v_end_t, logits=y_pred['v_end_logits'])
        )

        # J boundaries
        j_start_t = soft_targets(y_true['j_start'], L)
        j_end_t = soft_targets(y_true['j_end'], L)
        j_start_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=j_start_t, logits=y_pred['j_start_logits'])
        )
        j_end_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=j_end_t, logits=y_pred['j_end_logits'])
        )

        # Dynamic weighting per boundary
        weighted_v_start = v_start_loss * self.log_var_v_start(v_start_loss)
        weighted_v_end = v_end_loss * self.log_var_v_end(v_end_loss)
        weighted_j_start = j_start_loss * self.log_var_j_start(j_start_loss)
        weighted_j_end = j_end_loss * self.log_var_j_end(j_end_loss)

        segmentation_loss = weighted_v_start + weighted_v_end + weighted_j_start + weighted_j_end

        # D boundaries (optional)
        if self.has_d_gene:
            d_start_t = soft_targets(y_true['d_start'], L)
            d_end_t = soft_targets(y_true['d_end'], L)
            d_start_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=d_start_t, logits=y_pred['d_start_logits'])
            )
            d_end_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=d_end_t, logits=y_pred['d_end_logits'])
            )
            weighted_d_start = d_start_loss * self.log_var_d_start(d_start_loss)
            weighted_d_end = d_end_loss * self.log_var_d_end(d_end_loss)
            segmentation_loss += (weighted_d_start + weighted_d_end)

        # --- Auxiliary Segmentation Losses ---
        huber = keras.losses.Huber(delta=1.0)

        # Expectations for lengths
        v_s_exp = expectation_from_logits(y_pred['v_start_logits'])
        v_e_exp = expectation_from_logits(y_pred['v_end_logits'])
        j_s_exp = expectation_from_logits(y_pred['j_start_logits'])
        j_e_exp = expectation_from_logits(y_pred['j_end_logits'])

        v_len_pred = v_e_exp - v_s_exp
        j_len_pred = j_e_exp - j_s_exp
        v_len_true = tf.cast(y_true['v_end'], tf.float32) - tf.cast(y_true['v_start'], tf.float32)
        j_len_true = tf.cast(y_true['j_end'], tf.float32) - tf.cast(y_true['j_start'], tf.float32)

        len_loss = huber(v_len_true, tf.squeeze(v_len_pred, axis=-1)) + huber(j_len_true, tf.squeeze(j_len_pred, axis=-1))

        # IoU loss (1 - IoU) for intervals using expectations
        def interval_iou_loss(s_pred, e_pred, s_true, e_true, eps=1e-6):
            s_pred = tf.squeeze(s_pred, -1)
            e_pred = tf.squeeze(e_pred, -1)
            inter = tf.nn.relu(tf.minimum(e_pred, e_true) - tf.maximum(s_pred, s_true))
            len_pred = tf.maximum(e_pred - s_pred, 0.0)
            len_true = tf.maximum(e_true - s_true, 0.0)
            union = len_pred + len_true - inter + eps
            iou = inter / union
            return 1.0 - tf.reduce_mean(iou)

        iou_loss = interval_iou_loss(v_s_exp, v_e_exp, tf.cast(y_true['v_start'], tf.float32), tf.cast(y_true['v_end'], tf.float32)) \
                   + interval_iou_loss(j_s_exp, j_e_exp, tf.cast(y_true['j_start'], tf.float32), tf.cast(y_true['j_end'], tf.float32))

        # Hinge margin to keep spans at least 1 nt
        hinge_loss = tf.reduce_mean(tf.nn.relu(1.0 - tf.squeeze(v_len_pred, -1))) + \
                     tf.reduce_mean(tf.nn.relu(1.0 - tf.squeeze(j_len_pred, -1)))

        if self.has_d_gene:
            d_s_exp = expectation_from_logits(y_pred['d_start_logits'])
            d_e_exp = expectation_from_logits(y_pred['d_end_logits'])
            d_len_pred = d_e_exp - d_s_exp
            d_len_true = tf.cast(y_true['d_end'], tf.float32) - tf.cast(y_true['d_start'], tf.float32)
            len_loss += huber(d_len_true, tf.squeeze(d_len_pred, -1))
            iou_loss += interval_iou_loss(d_s_exp, d_e_exp, tf.cast(y_true['d_start'], tf.float32), tf.cast(y_true['d_end'], tf.float32))
            hinge_loss += tf.reduce_mean(tf.nn.relu(1.0 - tf.squeeze(d_len_pred, -1)))

        # Small weights for auxiliary terms
        aux_loss = 0.1 * len_loss + 0.1 * iou_loss + 0.05 * hinge_loss
        segmentation_loss += aux_loss

        # --- Classification Loss ---
        clf_v_loss = self._bce_loss_fn(y_true['v_allele'], y_pred['v_allele'])
        clf_j_loss = self._bce_loss_fn(y_true['j_allele'], y_pred['j_allele'])

        # Apply dynamic weighting
        weighted_v_clf_loss = clf_v_loss * self.log_var_v_classification(clf_v_loss)
        weighted_j_clf_loss = clf_j_loss * self.log_var_j_classification(clf_j_loss)

        classification_loss = weighted_v_clf_loss + weighted_j_clf_loss

        if self.has_d_gene:
            clf_d_loss = self._bce_loss_fn(y_true['d_allele'], y_pred['d_allele'])
            weighted_d_clf_loss = clf_d_loss * self.log_var_d_classification(clf_d_loss)
            classification_loss += weighted_d_clf_loss

            # Custom penalty for predicting short D-segments with high "short D" likelihood
            d_length_pred = y_pred['d_end'] - y_pred['d_start']
            short_d_prob = y_pred['d_allele'][:, -1]
            short_d_length_penalty = tf.reduce_mean(tf.cast(d_length_pred < 5, tf.float32) * short_d_prob)
            classification_loss += short_d_length_penalty

        # --- Analysis Losses (Mutation, Indel, Productivity) ---
        # Keras 3 removed many legacy functional aliases; compute losses explicitly
        mutation_rate_loss = tf.reduce_mean(tf.abs(tf.cast(y_true['mutation_rate'], tf.float32) - tf.cast(y_pred['mutation_rate'], tf.float32)))
        indel_count_loss = tf.reduce_mean(tf.abs(tf.cast(y_true['indel_count'], tf.float32) - tf.cast(y_pred['indel_count'], tf.float32)))
        productive_loss = keras.losses.BinaryCrossentropy()(y_true['productive'], y_pred['productive'])

        # Apply dynamic weighting
        weighted_mutation_loss = mutation_rate_loss * self.log_var_mutation(mutation_rate_loss)
        weighted_indel_loss = indel_count_loss * self.log_var_indel(indel_count_loss)
        weighted_productive_loss = productive_loss * self.log_var_productivity(productive_loss)

        # --- Total Loss ---
        total_loss = segmentation_loss + classification_loss + weighted_mutation_loss + weighted_indel_loss + weighted_productive_loss

        return total_loss, classification_loss, weighted_indel_loss, weighted_mutation_loss, segmentation_loss, weighted_productive_loss

    def train_step(self, data):
        """
        Custom training step.
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            losses = self.hierarchical_loss(y, y_pred)
            total_loss = losses[0]

        # Compute and apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update custom trackers only (avoid Keras 3 compiled_metrics internals)
        self.log_metrics(*losses)
        self.update_custom_trackers(y, y_pred)

        return self.get_metrics_log()

    def test_step(self, data):
        """
        Custom evaluation step.
        """
        x, y = data
        y_pred = self(x, training=False)
        losses = self.hierarchical_loss(y, y_pred)

        # Update custom trackers only
        self.log_metrics(*losses)
        self.update_custom_trackers(y, y_pred)

        return self.get_metrics_log()

    def log_metrics(self, total_loss, scaled_classification_loss, scaled_indel_count_loss,
                    scaled_mutation_rate_loss, segmentation_loss, scaled_productive_loss):
        """Updates the state of custom loss trackers."""
        self.loss_tracker.update_state(total_loss)
        self.scaled_classification_loss_tracker.update_state(scaled_classification_loss)
        self.scaled_indel_count_loss_tracker.update_state(scaled_indel_count_loss)
        self.scaled_mutation_rate_loss_tracker.update_state(scaled_mutation_rate_loss)
        self.segmentation_loss_tracker.update_state(segmentation_loss)
        self.scaled_productivity_loss_tracker.update_state(scaled_productive_loss)

    def update_custom_trackers(self, y_true, y_pred):
        """Updates custom monitoring trackers like entropy."""
        self.average_last_label_tracker.update_state(y_true, y_pred)
        self.v_allele_entropy_tracker.update_state(y_true, y_pred)
        self.j_allele_entropy_tracker.update_state(y_true, y_pred)
        if self.has_d_gene:
            self.d_allele_entropy_tracker.update_state(y_true, y_pred)

        # --- Boundary accuracy metrics ---
        def boundary_metrics(gt_scalar, logits):
            # gt: (B,1) float -> int indices via round
            gt_idx = tf.cast(tf.round(tf.squeeze(gt_scalar, axis=-1)), tf.int32)
            gt_idx = tf.clip_by_value(gt_idx, 0, self.max_seq_length - 1)
            pred_idx = tf.argmax(logits, axis=-1, output_type=tf.int32)
            err = tf.cast(tf.abs(pred_idx - gt_idx), tf.float32)
            mae = tf.reduce_mean(err)
            acc = tf.reduce_mean(tf.cast(tf.equal(pred_idx, gt_idx), tf.float32))
            acc1 = tf.reduce_mean(tf.cast(err <= 1.0, tf.float32))
            return mae, acc, acc1

        # V
        v_s_mae, v_s_acc, v_s_acc1 = boundary_metrics(y_true['v_start'], y_pred['v_start_logits'])
        v_e_mae, v_e_acc, v_e_acc1 = boundary_metrics(y_true['v_end'], y_pred['v_end_logits'])
        self.v_start_mae.update_state(v_s_mae)
        self.v_end_mae.update_state(v_e_mae)
        self.v_start_acc.update_state(v_s_acc)
        self.v_end_acc.update_state(v_e_acc)
        self.v_start_acc_1nt.update_state(v_s_acc1)
        self.v_end_acc_1nt.update_state(v_e_acc1)

        # J
        j_s_mae, j_s_acc, j_s_acc1 = boundary_metrics(y_true['j_start'], y_pred['j_start_logits'])
        j_e_mae, j_e_acc, j_e_acc1 = boundary_metrics(y_true['j_end'], y_pred['j_end_logits'])
        self.j_start_mae.update_state(j_s_mae)
        self.j_end_mae.update_state(j_e_mae)
        self.j_start_acc.update_state(j_s_acc)
        self.j_end_acc.update_state(j_e_acc)
        self.j_start_acc_1nt.update_state(j_s_acc1)
        self.j_end_acc_1nt.update_state(j_e_acc1)

        # Allele AUC metrics (treat multi-class as multi-label with sigmoid outputs)
        try:
            self.v_allele_auc.update_state(y_true['v_allele'], y_pred['v_allele'])
            self.j_allele_auc.update_state(y_true['j_allele'], y_pred['j_allele'])
            if self.has_d_gene:
                self.d_allele_auc.update_state(y_true['d_allele'], y_pred['d_allele'])
        except Exception:
            # Be resilient to unexpected shapes
            pass

        # D (optional)
        if self.has_d_gene:
            d_s_mae, d_s_acc, d_s_acc1 = boundary_metrics(y_true['d_start'], y_pred['d_start_logits'])
            d_e_mae, d_e_acc, d_e_acc1 = boundary_metrics(y_true['d_end'], y_pred['d_end_logits'])
            self.d_start_mae.update_state(d_s_mae)
            self.d_end_mae.update_state(d_e_mae)
            self.d_start_acc.update_state(d_s_acc)
            self.d_end_acc.update_state(d_e_acc)
            self.d_start_acc_1nt.update_state(d_s_acc1)
            self.d_end_acc_1nt.update_state(d_e_acc1)

    def get_metrics_log(self):
        """Constructs the dictionary of metrics to be logged."""
        # Start with an empty dict; compiled metrics are handled by Keras internals in TF/Keras 3
        # and may not be directly accessible via a stable API. We log our custom trackers here.
        metrics = {}
        # Add our custom loss trackers
        metrics.update({
            "loss": self.loss_tracker.result(),
            "segmentation_loss": self.segmentation_loss_tracker.result(),
            "classification_loss": self.scaled_classification_loss_tracker.result(),
            "mutation_rate_loss": self.scaled_mutation_rate_loss_tracker.result(),
            "indel_count_loss": self.scaled_indel_count_loss_tracker.result(),
            'productive_loss': self.scaled_productivity_loss_tracker.result(),
            'average_last_label': self.average_last_label_tracker.result(),
            'v_allele_entropy': self.v_allele_entropy_tracker.result(),
            'j_allele_entropy': self.j_allele_entropy_tracker.result()
        })
        # Boundary metrics
        metrics.update({
            'v_start_mae': self.v_start_mae.result(),
            'v_end_mae': self.v_end_mae.result(),
            'v_start_acc': self.v_start_acc.result(),
            'v_end_acc': self.v_end_acc.result(),
            'v_start_acc_1nt': self.v_start_acc_1nt.result(),
            'v_end_acc_1nt': self.v_end_acc_1nt.result(),
            'j_start_mae': self.j_start_mae.result(),
            'j_end_mae': self.j_end_mae.result(),
            'j_start_acc': self.j_start_acc.result(),
            'j_end_acc': self.j_end_acc.result(),
            'j_start_acc_1nt': self.j_start_acc_1nt.result(),
            'j_end_acc_1nt': self.j_end_acc_1nt.result(),
            'v_allele_auc': self.v_allele_auc.result(),
            'j_allele_auc': self.j_allele_auc.result(),
        })
        if self.has_d_gene:
            metrics['d_allele_entropy'] = self.d_allele_entropy_tracker.result()
            metrics.update({
                'd_start_mae': self.d_start_mae.result(),
                'd_end_mae': self.d_end_mae.result(),
                'd_start_acc': self.d_start_acc.result(),
                'd_end_acc': self.d_end_acc.result(),
                'd_start_acc_1nt': self.d_start_acc_1nt.result(),
                'd_end_acc_1nt': self.d_end_acc_1nt.result(),
                'd_allele_auc': self.d_allele_auc.result(),
            })
        return metrics

    @property
    def metrics(self):
        """
        Lists all metrics tracked by the model.

        *** THIS IS THE CRITICAL FIX ***
        This now includes both the custom loss trackers AND the standard
        metrics passed to model.compile() (e.g., AUC, accuracy).
        """
        # Start with the custom loss trackers
        metric_list = [
            self.loss_tracker,
            self.segmentation_loss_tracker,
            self.scaled_classification_loss_tracker,
            self.scaled_mutation_rate_loss_tracker,
            self.scaled_indel_count_loss_tracker,
            self.scaled_productivity_loss_tracker,
            self.average_last_label_tracker,
            self.v_allele_entropy_tracker,
            self.j_allele_entropy_tracker,
            self.v_allele_auc,
            self.j_allele_auc,
            # Boundary trackers
            self.v_start_mae,
            self.v_end_mae,
            self.v_start_acc,
            self.v_end_acc,
            self.v_start_acc_1nt,
            self.v_end_acc_1nt,
            self.j_start_mae,
            self.j_end_mae,
            self.j_start_acc,
            self.j_end_acc,
            self.j_start_acc_1nt,
            self.j_end_acc_1nt,
        ]
        if self.has_d_gene:
            metric_list.append(self.d_allele_entropy_tracker)
            metric_list += [
                self.d_start_mae,
                self.d_end_mae,
                self.d_start_acc,
                self.d_end_acc,
                self.d_start_acc_1nt,
                self.d_end_acc_1nt,
                self.d_allele_auc,
            ]

        # Do not append internal compiled metrics container; Keras 3 no longer exposes a stable list here.

        return metric_list

    def get_latent_representation(self, inputs, gene_type: str):
        """
        Extracts the latent representation for a specific gene (V, D, or J).

        Args:
            inputs (dict): Dictionary with 'tokenized_sequence'.
            gene_type (str): The gene to process. Must be 'V', 'D', or 'J'.

        Returns:
            tf.Tensor: The latent representation tensor before the final classification layer.
        """
        if gene_type.upper() not in ['V', 'D', 'J']:
            raise ValueError("gene_type must be 'V', 'D', or 'J'.")
        if gene_type.upper() == 'D' and not self.has_d_gene:
            raise ValueError("Cannot get D-gene representation; model configured without D-gene.")

        # Common initial steps
        input_seq = tf.reshape(inputs["tokenized_sequence"], (-1, self.max_seq_length))
        input_seq = tf.cast(input_seq, "float32")
        input_embeddings = self.input_embeddings(input_seq)

        # Gene-specific path
        positions = tf.cast(tf.range(self.max_seq_length), tf.float32)[tf.newaxis, :]
        if gene_type.upper() == 'V':
            segment_features = self.v_segmentation_feature_block(input_embeddings)
            s_logits = self.v_start_head(segment_features)
            e_logits = self.v_end_head(segment_features)
            s_exp = tf.reduce_sum(tf.nn.softmax(s_logits, -1) * positions, axis=-1, keepdims=True)
            e_exp = tf.reduce_sum(tf.nn.softmax(e_logits, -1) * positions, axis=-1, keepdims=True)
            mask = self.v_mask_layer([s_exp, e_exp])
            mask_reshape = self.v_mask_reshape(mask)
            masked_sequence = self.v_mask_gate([input_embeddings, mask_reshape])
            feature_map = self.v_feature_extraction_block(masked_sequence)
            latent_rep = self.v_allele_mid(feature_map)
        elif gene_type.upper() == 'J':
            segment_features = self.j_segmentation_feature_block(input_embeddings)
            s_logits = self.j_start_head(segment_features)
            e_logits = self.j_end_head(segment_features)
            s_exp = tf.reduce_sum(tf.nn.softmax(s_logits, -1) * positions, axis=-1, keepdims=True)
            e_exp = tf.reduce_sum(tf.nn.softmax(e_logits, -1) * positions, axis=-1, keepdims=True)
            mask = self.j_mask_layer([s_exp, e_exp])
            mask_reshape = self.j_mask_reshape(mask)
            masked_sequence = self.j_mask_gate([input_embeddings, mask_reshape])
            feature_map = self.j_feature_extraction_block(masked_sequence)
            latent_rep = self.j_allele_mid(feature_map)
        else: # D-gene
            segment_features = self.d_segmentation_feature_block(input_embeddings)
            s_logits = self.d_start_head(segment_features)
            e_logits = self.d_end_head(segment_features)
            s_exp = tf.reduce_sum(tf.nn.softmax(s_logits, -1) * positions, axis=-1, keepdims=True)
            e_exp = tf.reduce_sum(tf.nn.softmax(e_logits, -1) * positions, axis=-1, keepdims=True)
            mask = self.d_mask_layer([s_exp, e_exp])
            mask_reshape = self.d_mask_reshape(mask)
            masked_sequence = self.d_mask_gate([input_embeddings, mask_reshape])
            feature_map = self.d_feature_extraction_block(masked_sequence)
            latent_rep = self.d_allele_mid(feature_map)

        return latent_rep

    def serialization_config(self):
        """Return a dict conforming to ModelBundleConfig fields for this instance."""
        return {
            'model_type': 'single_chain',
            'format_version': FORMAT_VERSION,
            'max_seq_length': self.max_seq_length,
            'has_d_gene': self.has_d_gene,
            'v_allele_count': self.v_allele_count,
            'j_allele_count': self.j_allele_count,
            'd_allele_count': getattr(self, 'd_allele_count', None) if self.has_d_gene else None,
            'v_allele_latent_size': self.v_allele_latent_size,
            'j_allele_latent_size': self.j_allele_latent_size,
            'd_allele_latent_size': getattr(self, 'd_allele_latent_size', None) if self.has_d_gene else None,
            'chain_types': None,
            'number_of_chains': None,
        }

    def save_pretrained(self, bundle_dir: str | os.PathLike, training_meta: TrainingMeta | None = None,
                        saved_model_subdir: str = 'saved_model',
                        include_logits_in_saved_model: bool = True,
                        include_keras_weights_checkpoint: bool = True):
        """Save a SavedModel + structural config + dataconfig into a versioned bundle.

        Parameters
        ----------
        bundle_dir : Path-like
            Target directory to create / overwrite bundle contents.
        training_meta : TrainingMeta, optional
            Optional training metadata. If not provided a minimal placeholder is written.
        include_logits_in_saved_model : bool
            If True, append raw boundary logits tensors to the SavedModel signature outputs.
        include_keras_weights_checkpoint : bool
            If True, also write a trainable Keras checkpoint file 'checkpoint.weights.h5' into the bundle
            for advanced fine-tuning workflows.
        """
        bundle_path = Path(bundle_dir)
        # Ensure model is built
        if not self.built:
            dummy = {"tokenized_sequence": tf.zeros((1, self.max_seq_length), dtype=tf.float32)}
            _ = self(dummy, training=False)
        bundle_path.mkdir(parents=True, exist_ok=True)
        cfg = ModelBundleConfig(**(self.serialization_config()))
        # Enrich config with environment info
        try:
            import tensorflow as _tf
            cfg.tf_version = _tf.__version__
        except Exception:  # pragma: no cover
            pass
        if training_meta is None:
            training_meta = TrainingMeta(
                epochs_trained=0,
                final_epoch=0,
                best_epoch=None,
                best_loss=None,
                final_loss=None,
                metrics_summary={},
            )
        # Persist non-weight artifacts
        save_bundle(bundle_path, cfg, self.dataconfig, training_meta)
        # Always export a TensorFlow SavedModel for robust deployment/loading
        self.export_saved_model(bundle_path / saved_model_subdir, include_logits=include_logits_in_saved_model)
        
        # Optionally persist a trainable Keras checkpoint for fine-tuning
        if include_keras_weights_checkpoint:
            try:
                ckpt_path = bundle_path / "checkpoint.weights.h5"
                # Keras 3 HDF5 weights checkpoint (trainable)
                self.save_weights(ckpt_path.as_posix())
                logger.info("Saved Keras weights checkpoint to %s", ckpt_path)
            except Exception:
                logger.warning("Failed to save Keras weights checkpoint; proceeding without it.", exc_info=True)
        # Recompute fingerprint to include SavedModel assets (weights checkpoint is intentionally excluded)
        try:
            from AlignAIR.Serialization.validators import compute_fingerprint as _cfp
            (bundle_path / "fingerprint.txt").write_text(_cfp(bundle_path))
        except Exception:
            pass
        logger.info("Saved pretrained bundle to %s", bundle_path)

    @classmethod
    def from_pretrained(cls, bundle_dir: str | os.PathLike):
        """Load a model from a SavedModel-first bundle. H5 is no longer supported."""
        bundle_path = Path(bundle_dir)
        cfg, dataconfig_obj, _meta = load_bundle(bundle_path)

        sm_dir = bundle_path / 'saved_model'
        if sm_dir.exists():
            wrapper = SavedModelInferenceWrapper(
                saved_model_dir=sm_dir,
                bundle_dir=bundle_path,
                config=cfg.__dict__ if hasattr(cfg, '__dict__') else None,
            )
            wrapper.dataconfig = dataconfig_obj
            logger.info("Loaded SavedModel from %s", sm_dir)
            return wrapper
        raise FileNotFoundError(f"SavedModel not found in bundle: {sm_dir}. This AlignAIRR version requires SavedModel-first bundles.")

    # ---------------- SavedModel Export (Step 8) -----------------
    def export_saved_model(self, export_dir: Union[str, os.PathLike], include_logits: bool = False):
        """Export a TF SavedModel for serving.

        Parameters
        ----------
        export_dir : path-like
            Target directory for the SavedModel (will be created / overwritten).
        include_logits : bool
            If True, include the raw *_start_logits / *_end_logits tensors in the exported signature.
        """
        export_path = Path(export_dir)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure model is built
        if not self.built:
            dummy = {"tokenized_sequence": tf.zeros((1, self.max_seq_length), dtype=tf.int32)}
            _ = self(dummy, training=False)

        # Export a lightweight inference module with a concrete signature for robustness
        class _InferenceModule(tf.Module):  # pragma: no cover - export graph
            def __init__(self, model, max_len: int, include_logits_flag: bool):
                super().__init__()
                self.model = model
                self.L = int(max_len)
                self.include_logits_flag = bool(include_logits_flag)

                def _serving(tokenized_sequence):
                    x = tf.convert_to_tensor(tokenized_sequence)
                    if x.dtype not in (tf.int32, tf.int64):
                        x = tf.cast(x, tf.int32)
                    outputs = self.model({'tokenized_sequence': x}, training=False)
                    allowed_keys = [
                        'v_start', 'v_end', 'j_start', 'j_end',
                        'v_allele', 'j_allele', 'mutation_rate', 'indel_count', 'productive'
                    ]
                    if 'd_start' in outputs:
                        allowed_keys += ['d_start', 'd_end', 'd_allele']
                    if self.include_logits_flag:
                        allowed_keys += [k for k in outputs.keys() if k.endswith('_logits')]
                    return {k: tf.identity(outputs[k], name=k) for k in allowed_keys if k in outputs}

                # Create a tf.function with a concrete input signature
                self.serving_default = tf.function(
                    _serving,
                    input_signature=[tf.TensorSpec(shape=[None, self.L], dtype=tf.int32, name='tokenized_sequence')]
                )

        module = _InferenceModule(self, self.max_seq_length, include_logits)
        # Let TF create the ConcreteFunction automatically from the tf.function
        tf.saved_model.save(module, str(export_path), signatures={'serving_default': module.serving_default})
        logger.info("Exported SavedModel to %s", export_path)
