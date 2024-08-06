import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Multiply,
    Reshape,
    Flatten
)
from ..HeavyChain.losses import d_loss
from ..Layers import ConvResidualFeatureExtractionBlock, RegularizedConstrainedLogVar, CutoutLayer, \
    Conv1D_and_BatchNorm, EntropyMetric
from ..Layers import (
    TokenAndPositionEmbedding, MinMaxValueConstraint
)




class LightChainAlignAIRR(tf.keras.Model):
    """
      The AlignAIRR model for performing segmentation, mutation rate estimation,
      and allele classification tasks in heavy chain sequences.

      Attributes:
          max_seq_length (int): Maximum sequence length.
          v_allele_count (int): Number of V alleles.
          j_allele_count (int): Number of J alleles.
          ... (other attributes)
      """

    def __init__(self, max_seq_length, v_allele_count, j_allele_count):
        super(LightChainAlignAIRR, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.GlorotUniform()  # RandomNormal(mean=0.1, stddev=0.02)

        # Model Params
        self.max_seq_length = int(max_seq_length)
        self.v_allele_count = v_allele_count
        self.j_allele_count = j_allele_count

        # Hyperparams + Constants
        self.classification_keys = ["v_allele", "j_allele"]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.v_class_weight, self.j_class_weight = 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (0.5, 0.5, 0.5)

        self.BCE_SMOOTH = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # Tracking
        self.setup_performance_metrics()
        self.setup_model()
        # Define task-specific log variances for dynamic weighting
        self.setup_log_variances()

    def setup_model(self):
        # Init Input Layers
        self._init_input_layers()

        self.input_embeddings = TokenAndPositionEmbedding(
            vocab_size=6, embed_dim=32, maxlen=self.max_seq_length
        )
        self.fblock_activation = 'tanh'
        print(self.fblock_activation)

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self.meta_feature_extractor_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                               num_conv_batch_layers=4,
                                                                               kernel_size=[3, 3, 3, 2, 5],
                                                                               max_pool_size=2,
                                                                               conv_activation=tf.keras.layers.Activation(
                                                                                   self.fblock_activation),
                                                                               initializer=self.initializer)

        self.v_segmentation_feature_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                               num_conv_batch_layers=4,
                                                                               kernel_size=[3, 3, 3, 2, 5],
                                                                               max_pool_size=2,
                                                                               conv_activation=tf.keras.layers.Activation(
                                                                                   self.fblock_activation),
                                                                               initializer=self.initializer)

        self.j_segmentation_feature_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                               num_conv_batch_layers=4,
                                                                               kernel_size=[3, 3, 3, 2, 5],
                                                                               max_pool_size=2,
                                                                               conv_activation=tf.keras.layers.Activation(
                                                                                   self.fblock_activation),
                                                                               initializer=self.initializer)

        self.v_mask_layer = CutoutLayer(gene='V', max_size=self.max_seq_length)
        self.j_mask_layer = CutoutLayer(gene='J', max_size=self.max_seq_length)

        # Init V/D/J Masked Input Signal Encoding Layers
        # Init V/D/J Masked Input Signal Encoding Layers
        self.v_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                             num_conv_batch_layers=6,
                                                                             kernel_size=[3, 3, 3, 2, 2, 2, 5],
                                                                             max_pool_size=2,
                                                                             conv_activation=tf.keras.layers.Activation(
                                                                                 self.fblock_activation),
                                                                             initializer=self.initializer)

        self.j_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                             num_conv_batch_layers=6,
                                                                             kernel_size=[3, 3, 3, 2, 2, 2, 5],
                                                                             max_pool_size=2,
                                                                             conv_activation=tf.keras.layers.Activation(
                                                                                 self.fblock_activation),
                                                                             initializer=self.initializer)

        # Init Interval Regression Related Layers
        self._init_segmentation_predictions()

        # Init the masking layer that will leverage the predicted segmentation mask
        self._init_masking_layers()

        #  =========== V HEADS ======================
        self._init_v_classification_layers()

        # =========== J HEADS ======================
        self._init_j_classification_layers()

    def setup_log_variances(self):
        """Initialize log variances for dynamic weighting."""
        self.log_var_v_start = RegularizedConstrainedLogVar()
        self.log_var_v_end = RegularizedConstrainedLogVar()
        self.log_var_j_start = RegularizedConstrainedLogVar()
        self.log_var_j_end = RegularizedConstrainedLogVar()
        self.log_var_v_classification = RegularizedConstrainedLogVar()
        self.log_var_j_classification = RegularizedConstrainedLogVar()
        self.log_var_mutation = RegularizedConstrainedLogVar()
        self.log_var_indel = RegularizedConstrainedLogVar()
        self.log_var_productivity = RegularizedConstrainedLogVar()
        self.log_var_chain_type = RegularizedConstrainedLogVar()

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

        # Add custom metrics for monitoring
        self.v_allele_entropy_tracker = EntropyMetric(allele_name="v_allele")
        self.j_allele_entropy_tracker = EntropyMetric(allele_name="j_allele")

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, "float32")
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name="seq_init")

    def _init_masking_layers(self):
        self.v_mask_gate = Multiply()
        self.v_mask_reshape = Reshape((self.max_seq_length, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((self.max_seq_length, 1))

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle", kernel_initializer=self.initializer,
        )

        self.v_allele_call_head = Dense(self.v_allele_count, activation="sigmoid", name="v_allele")

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

        self.chain_type_mid = Dense(self.max_seq_length, activation=act, name="chain_type_mid",
                                    kernel_initializer=self.initializer
                                    )
        self.chain_type_dropout = Dropout(0.05)
        self.chain_type_head = Dense(1, activation='sigmoid', name="chain_type", kernel_initializer=self.initializer)

        self.productivity_feature_block = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=1,
                                                               initializer=self.initializer)
        self.productivity_flatten = Flatten()

        self.productivity_dropout = Dropout(0.05)
        self.productivity_head = Dense(
            1, activation='sigmoid', name="productive", kernel_initializer=self.initializer
        )

    def _predict_vdj_set(self, v_feature_map, j_feature_map):
        # ============================ V =============================
        v_allele_middle = self.v_allele_mid(v_feature_map)
        v_allele = self.v_allele_call_head(v_allele_middle)

        # ============================ J =============================
        j_allele_middle = self.j_allele_mid(j_feature_map)
        j_allele = self.j_allele_call_head(j_allele_middle)

        return v_allele, j_allele

    def mask_embeddings(self, input_embeddings, end_positions):
        """
        Masks the input embeddings up to the given end positions.

        Args:
        - input_embeddings: Tensor of shape (batch_size, seq_length, embedding_dim)
        - end_positions: Tensor of shape (batch_size, 1) with the end positions to mask up to

        Returns:
        - masked_embeddings: Tensor of the same shape as input_embeddings with values up to end_positions masked to zero
        """
        batch_size, seq_length, embedding_dim = tf.shape(input_embeddings)[0], tf.shape(input_embeddings)[1], \
        tf.shape(input_embeddings)[2]

        # Create a mask for each position in the batch
        mask = tf.sequence_mask(end_positions, seq_length, dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)  # Shape: (batch_size, seq_length, 1)
        mask = tf.tile(mask, [1, 1, embedding_dim])  # Shape: (batch_size, seq_length, embedding_dim)

        # Apply the mask to the input embeddings
        masked_embeddings = input_embeddings * mask
        return masked_embeddings

    def call(self, inputs):
        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        meta_features = self.meta_feature_extractor_block(input_embeddings)
        v_segment_features = self.v_segmentation_feature_block(input_embeddings)
        j_segment_features = self.j_segmentation_feature_block(input_embeddings)

        v_start = self.v_start_out(v_segment_features)
        v_end = self.v_end_out(v_segment_features)

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
        reshape_masked_sequence_j = self.j_mask_layer([j_start, j_end])

        reshape_masked_sequence_v = tf.expand_dims(reshape_masked_sequence_v, -1)
        reshape_masked_sequence_j = tf.expand_dims(reshape_masked_sequence_j, -1)

        chain_type_mid = self.chain_type_mid(meta_features)
        chain_type_mid = self.chain_type_dropout(chain_type_mid)
        chain_type = self.chain_type_head(chain_type_mid)

        masked_sequence_v = self.v_mask_gate([input_embeddings, reshape_masked_sequence_v])
        masked_sequence_j = self.j_mask_gate([input_embeddings, reshape_masked_sequence_j])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self.v_feature_extraction_block(masked_sequence_v)
        j_feature_map = self.j_feature_extraction_block(masked_sequence_j)

        # STEP 8: Predict The V,D and J genes
        v_allele, j_allele = self._predict_vdj_set(v_feature_map, j_feature_map)

        return {
            "v_start": v_start,
            "v_end": v_end,
            "j_start": j_start,
            "j_end": j_end,
            "v_allele": v_allele,
            "j_allele": j_allele,
            'mutation_rate': mutation_rate,
            'indel_count': indel_count,
            'productive': is_productive,
            'type': chain_type
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

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, "float32")

    def hierarchical_loss(self, y_true, y_pred):
        # Extract the segmentation and classification outputs
        # segmentation_true = [self.c2f32(y_true[k]) for k in ['v_segment', 'd_segment', 'j_segment']]
        # segmentation_pred = [self.c2f32(y_pred[k]) for k in ['v_segment', 'd_segment', 'j_segment']]

        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        # Segmentation Loss
        v_start_loss = tf.keras.losses.mean_absolute_error(y_true['v_start'], y_pred['v_start'])
        v_end_loss = tf.keras.losses.mean_absolute_error(y_true['v_end'], y_pred['v_end'])

        j_start_loss = tf.keras.losses.mean_absolute_error(y_true['j_start'], y_pred['j_start'])
        j_end_loss = tf.keras.losses.mean_absolute_error(y_true['j_end'], y_pred['j_end'])

        # Calculate the precision as the inverse of the exponential of each task's log variance
        weighted_v_start = v_start_loss * self.log_var_v_start(v_start_loss)
        weighted_v_end = v_end_loss * self.log_var_v_end(v_end_loss)

        weighted_j_start = j_start_loss * self.log_var_j_start(j_start_loss)
        weighted_j_end = j_end_loss * self.log_var_j_end(j_end_loss)

        ##########################################################################################################

        segmentation_loss = weighted_v_start + weighted_v_end + weighted_j_start + weighted_j_end

        # Classification Loss

        clf_v_loss = tf.keras.losses.binary_crossentropy(classification_true[0], classification_pred[0])
        clf_j_loss = tf.keras.losses.binary_crossentropy(classification_true[1], classification_pred[1])

        precision_v_classification = self.log_var_v_classification(clf_v_loss)
        precision_j_classification = self.log_var_j_classification(clf_j_loss)
        classification_loss = precision_v_classification * clf_v_loss + precision_j_classification * clf_j_loss

        # Mutation Loss
        mutation_rate_loss = tf.keras.losses.mean_absolute_error(y_true['mutation_rate'], y_pred['mutation_rate'])
        # Indel Count Loss
        indel_count_loss = tf.keras.losses.mean_absolute_error(y_true['indel_count'], y_pred['indel_count'])

        # Compute Productivity Loss
        productive_loss = tf.keras.losses.binary_crossentropy(y_true['productive'], y_pred['productive'])

        chain_type_loss = tf.keras.metrics.binary_crossentropy(self.c2f32(y_true['type']), self.c2f32(y_pred['type']))

        precision_mutation = self.log_var_mutation(mutation_rate_loss)
        precision_indel = self.log_var_indel(indel_count_loss)
        precision_productivity = self.log_var_productivity(productive_loss)
        precision_type = self.log_var_chain_type(chain_type_loss)

        mutation_rate_loss *= precision_mutation
        indel_count_loss *= precision_indel
        productive_loss *= precision_productivity
        chain_type_loss *= precision_type

        # Get precision (inverse of variance) for each task

        # Sum weighted losses
        total_loss = (segmentation_loss + classification_loss +
                      mutation_rate_loss + indel_count_loss +
                      productive_loss + chain_type_loss)

        return total_loss, classification_loss, indel_count_loss, mutation_rate_loss, segmentation_loss, productive_loss

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
        # Update custom metrics
        self.v_allele_entropy_tracker.update_state(y, y_pred)
        self.j_allele_entropy_tracker.update_state(y, y_pred)

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
        metrics['v_allele_entropy'] = self.v_allele_entropy_tracker.result()
        metrics['j_allele_entropy'] = self.j_allele_entropy_tracker.result()
        return metrics

    def model_summary(self, input_shape):
        x = {
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

