import numpy as np
import tensorflow as tf
from GenAIRR.dataconfig import DataConfig
from tensorflow.keras import Model, regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Multiply,
    Reshape,
    Flatten,
    Activation
)

# Assuming these are custom layers from your project structure.
# Ensure the relative paths are correct for your environment.
from ..Layers import Conv1D_and_BatchNorm, CutoutLayer, AverageLastLabel, EntropyMetric
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
                 v_allele_latent_size: int = None,
                 d_allele_latent_size: int = None,
                 j_allele_latent_size: int = None):
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
        self.initializer = tf.keras.initializers.GlorotUniform()

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
        self._bce_loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # --- Model Setup ---
        self.setup_model_layers()
        self.setup_log_variances()
        self.setup_performance_metrics()

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
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.scaled_classification_loss_tracker = tf.keras.metrics.Mean(name="scaled_classification_loss")
        self.scaled_indel_count_loss_tracker = tf.keras.metrics.Mean(name="scaled_indel_count_loss")
        self.scaled_productivity_loss_tracker = tf.keras.metrics.Mean(name="scaled_productivity_loss")
        self.scaled_mutation_rate_loss_tracker = tf.keras.metrics.Mean(name="scaled_mutation_rate_loss")
        self.segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")

        # Custom metrics for deeper monitoring
        self.average_last_label_tracker = AverageLastLabel(name="average_last_label")
        self.v_allele_entropy_tracker = EntropyMetric(allele_name="v_allele")
        self.j_allele_entropy_tracker = EntropyMetric(allele_name="j_allele")

        if self.has_d_gene:
            self.d_allele_entropy_tracker = EntropyMetric(allele_name="d_allele")

    def _init_input_and_embedding_layers(self):
        """Initializes input and embedding layers."""
        self.input_layer = Input((self.max_seq_length, 1), name="seq_init")
        self.input_embeddings = TokenAndPositionEmbedding(
            vocab_size=6, embed_dim=32, maxlen=self.max_seq_length
        )

    def _init_feature_extractors(self):
        """Initializes convolutional feature extraction blocks."""
        conv_activation = Activation(self.fblock_activation)

        # Shared block for meta features (mutation, indel, productivity)
        self.meta_feature_extractor_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer
        )

        # Blocks for segmentation feature extraction
        self.v_segmentation_feature_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer
        )
        self.j_segmentation_feature_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer
        )

        # Blocks for classification feature extraction (on masked sequences)
        self.v_feature_extraction_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=6, kernel_size=[3, 3, 3, 2, 2, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer
        )
        self.j_feature_extraction_block = ConvResidualFeatureExtractionBlock(
            filter_size=128, num_conv_batch_layers=6, kernel_size=[3, 3, 3, 2, 2, 2, 5],
            max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer
        )

        if self.has_d_gene:
            self.d_segmentation_feature_block = ConvResidualFeatureExtractionBlock(
                filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 3, 2, 5],
                max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer
            )
            self.d_feature_extraction_block = ConvResidualFeatureExtractionBlock(
                filter_size=128, num_conv_batch_layers=4, kernel_size=[3, 3, 2, 2, 5],
                max_pool_size=2, conv_activation=conv_activation, initializer=self.initializer
            )

    def _init_segmentation_heads(self):
        """Initializes the output layers for segment boundaries."""
        act = tf.keras.activations.gelu
        constraint = unit_norm()

        self.v_start_out = Dense(1, activation=act, kernel_constraint=constraint, kernel_initializer=self.initializer, name='v_start')
        self.v_end_out = Dense(1, activation=act, kernel_constraint=constraint, kernel_initializer=self.initializer, name='v_end')
        self.j_start_out = Dense(1, activation=act, kernel_constraint=constraint, kernel_initializer=self.initializer, name='j_start')
        self.j_end_out = Dense(1, activation=act, kernel_constraint=constraint, kernel_initializer=self.initializer, name='j_end')

        if self.has_d_gene:
            self.d_start_out = Dense(1, activation=act, kernel_constraint=constraint, kernel_initializer=self.initializer, name='d_start')
            self.d_end_out = Dense(1, activation=act, kernel_constraint=constraint, kernel_initializer=self.initializer, name='d_end')

    def _init_analysis_heads(self):
        """Initializes heads for mutation rate, indel count, and productivity."""
        act = tf.keras.activations.gelu

        # Mutation Rate Head
        self.mutation_rate_mid = Dense(self.max_seq_length, activation=act, name="mutation_rate_mid", kernel_initializer=self.initializer)
        self.mutation_rate_dropout = Dropout(0.05)
        self.mutation_rate_head = Dense(1, activation='relu', name="mutation_rate", kernel_initializer=self.initializer, kernel_constraint=MinMaxValueConstraint(0, 1))

        # Indel Count Head
        self.indel_count_mid = Dense(self.max_seq_length, activation=act, name="indel_count_mid", kernel_initializer=self.initializer)
        self.indel_count_dropout = Dropout(0.05)
        self.indel_count_head = Dense(1, activation='relu', name="indel_count", kernel_initializer=self.initializer, kernel_constraint=MinMaxValueConstraint(0, 50))

        # Productivity Head
        self.productivity_flatten = Flatten()
        self.productivity_dropout = Dropout(0.05)
        self.productivity_head = Dense(1, activation='sigmoid', name="productive", kernel_initializer=self.initializer)

    def _init_classification_heads(self):
        """Initializes the output layers for allele classification."""
        # V-allele classification
        v_latent_dim = self.v_allele_latent_size or self.v_allele_count * self.latent_size_factor
        self.v_allele_mid = Dense(v_latent_dim, activation=self.classification_middle_layer_activation, name="v_allele_middle", kernel_initializer=self.initializer)
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
        self.v_mask_layer = CutoutLayer(gene='V', max_size=self.max_seq_length)
        self.j_mask_layer = CutoutLayer(gene='J', max_size=self.max_seq_length)

        self.v_mask_gate = Multiply()
        self.j_mask_gate = Multiply()

        self.v_mask_reshape = Reshape((self.max_seq_length, 1))
        self.j_mask_reshape = Reshape((self.max_seq_length, 1))

        if self.has_d_gene:
            self.d_mask_layer = CutoutLayer(gene='D', max_size=self.max_seq_length)
            self.d_mask_gate = Multiply()
            self.d_mask_reshape = Reshape((self.max_seq_length, 1))

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

        # 2. Predict Segmentation Boundaries
        v_start = self.v_start_out(v_segment_features)
        v_end = self.v_end_out(v_segment_features)
        j_start = self.j_start_out(j_segment_features)
        j_end = self.j_end_out(j_segment_features)

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
        v_mask = self.v_mask_layer([v_start, v_end])
        v_mask = self.v_mask_reshape(v_mask)
        masked_sequence_v = self.v_mask_gate([input_embeddings, v_mask])

        j_mask = self.j_mask_layer([j_start, j_end])
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
            d_start = self.d_start_out(d_segment_features)
            d_end = self.d_end_out(d_segment_features)

            d_mask = self.d_mask_layer([d_start, d_end])
            d_mask = self.d_mask_reshape(d_mask)
            masked_sequence_d = self.d_mask_gate([input_embeddings, d_mask])

            d_feature_map = self.d_feature_extraction_block(masked_sequence_d)
            d_allele_latent = self.d_allele_mid(d_feature_map)
            d_allele = self.d_allele_call_head(d_allele_latent)

        # 7. Compile final output dictionary
        output = {
            "v_start": v_start, "v_end": v_end,
            "j_start": j_start, "j_end": j_end,
            "v_allele": v_allele, "j_allele": j_allele,
            'mutation_rate': mutation_rate,
            'indel_count': indel_count,
            'productive': is_productive
        }
        if self.has_d_gene:
            output.update({'d_start': d_start, 'd_end': d_end, 'd_allele': d_allele})

        return output

    def hierarchical_loss(self, y_true, y_pred):
        """
        Calculates the total loss as a dynamically weighted sum of task-specific losses.
        """
        # --- Segmentation Loss ---
        v_start_loss = tf.keras.losses.mean_absolute_error(y_true['v_start'], y_pred['v_start'])
        v_end_loss = tf.keras.losses.mean_absolute_error(y_true['v_end'], y_pred['v_end'])
        j_start_loss = tf.keras.losses.mean_absolute_error(y_true['j_start'], y_pred['j_start'])
        j_end_loss = tf.keras.losses.mean_absolute_error(y_true['j_end'], y_pred['j_end'])

        # Apply dynamic weighting
        weighted_v_start = v_start_loss * self.log_var_v_start(v_start_loss)
        weighted_v_end = v_end_loss * self.log_var_v_end(v_end_loss)
        weighted_j_start = j_start_loss * self.log_var_j_start(j_start_loss)
        weighted_j_end = j_end_loss * self.log_var_j_end(j_end_loss)

        segmentation_loss = weighted_v_start + weighted_v_end + weighted_j_start + weighted_j_end

        if self.has_d_gene:
            d_start_loss = tf.keras.losses.mean_absolute_error(y_true['d_start'], y_pred['d_start'])
            d_end_loss = tf.keras.losses.mean_absolute_error(y_true['d_end'], y_pred['d_end'])
            weighted_d_start = d_start_loss * self.log_var_d_start(d_start_loss)
            weighted_d_end = d_end_loss * self.log_var_d_end(d_end_loss)
            segmentation_loss += (weighted_d_start + weighted_d_end)

        # --- Classification Loss ---
        clf_v_loss = tf.keras.losses.binary_crossentropy(y_true['v_allele'], y_pred['v_allele'])
        clf_j_loss = tf.keras.losses.binary_crossentropy(y_true['j_allele'], y_pred['j_allele'])

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
            short_d_prob = y_pred['d_allele'][:, -1] # Assuming last class is "short D"
            short_d_length_penalty = tf.reduce_mean(
                tf.cast(d_length_pred < 5, tf.float32) * short_d_prob
            )
            classification_loss += short_d_length_penalty

        # --- Analysis Losses (Mutation, Indel, Productivity) ---
        mutation_rate_loss = tf.keras.losses.mean_absolute_error(y_true['mutation_rate'], y_pred['mutation_rate'])
        indel_count_loss = tf.keras.losses.mean_absolute_error(y_true['indel_count'], y_pred['indel_count'])
        productive_loss = tf.keras.losses.binary_crossentropy(y_true['productive'], y_pred['productive'])

        # Apply dynamic weighting
        weighted_mutation_loss = mutation_rate_loss * self.log_var_mutation(mutation_rate_loss)
        weighted_indel_loss = indel_count_loss * self.log_var_indel(indel_count_loss)
        weighted_productive_loss = productive_loss * self.log_var_productivity(productive_loss)

        # --- Total Loss ---
        total_loss = (segmentation_loss + classification_loss +
                      weighted_mutation_loss + weighted_indel_loss +
                      weighted_productive_loss)

        return (total_loss, classification_loss, weighted_indel_loss,
                weighted_mutation_loss, segmentation_loss, weighted_productive_loss)

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

        # Update standard and custom metrics
        self.compiled_metrics.update_state(y, y_pred)
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

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
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

    def get_metrics_log(self):
        """Constructs the dictionary of metrics to be logged."""
        # Start with metrics from compile()
        metrics = {m.name: m.result() for m in self.compiled_metrics.metrics}  # Use self.compiled_metrics
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
        if self.has_d_gene:
            metrics['d_allele_entropy'] = self.d_allele_entropy_tracker.result()
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
        ]
        if self.has_d_gene:
            metric_list.append(self.d_allele_entropy_tracker)

        # Add the metrics from compile()
        if self.compiled_metrics is not None:
            metric_list += self.compiled_metrics.metrics

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
        if gene_type.upper() == 'V':
            segment_features = self.v_segmentation_feature_block(input_embeddings)
            start = self.v_start_out(segment_features)
            end = self.v_end_out(segment_features)
            mask = self.v_mask_layer([start, end])
            mask_reshape = self.v_mask_reshape(mask)
            masked_sequence = self.v_mask_gate([input_embeddings, mask_reshape])
            feature_map = self.v_feature_extraction_block(masked_sequence)
            latent_rep = self.v_allele_mid(feature_map)
        elif gene_type.upper() == 'J':
            segment_features = self.j_segmentation_feature_block(input_embeddings)
            start = self.j_start_out(segment_features)
            end = self.j_end_out(segment_features)
            mask = self.j_mask_layer([start, end])
            mask_reshape = self.j_mask_reshape(mask)
            masked_sequence = self.j_mask_gate([input_embeddings, mask_reshape])
            feature_map = self.j_feature_extraction_block(masked_sequence)
            latent_rep = self.j_allele_mid(feature_map)
        else: # D-gene
            segment_features = self.d_segmentation_feature_block(input_embeddings)
            start = self.d_start_out(segment_features)
            end = self.d_end_out(segment_features)
            mask = self.d_mask_layer([start, end])
            mask_reshape = self.d_mask_reshape(mask)
            masked_sequence = self.d_mask_gate([input_embeddings, mask_reshape])
            feature_map = self.d_feature_extraction_block(masked_sequence)
            latent_rep = self.d_allele_mid(feature_map)

        return latent_rep
