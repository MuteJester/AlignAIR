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
    Reshape
)
from Models.HeavyChain.losses import d_loss
from Models.Layers.Layers import ConvResidualFeatureExtractionBlock
from Models.Layers import (
    TokenAndPositionEmbedding, MinMaxValueConstraint
)


class HeavyChainAlignAIRR(tf.keras.Model):
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
        super(HcExperimental, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.GlorotUniform()#RandomNormal(mean=0.1, stddev=0.02)

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

        # Tracking
        self.setup_performance_metrics()

        self.setup_model()

    def setup_model(self):
        # Init Input Layers
        self._init_input_layers()

        self.input_embeddings = TokenAndPositionEmbedding(
            vocab_size=6, emded_dim=32, maxlen=self.max_seq_length
        )

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self.segmentation_feature_extractor_block = ConvResidualFeatureExtractionBlock(filter_size=64,
                                                                                       num_conv_batch_layers=5,
                                                                                       kernel_size=[2,5,5,3,5,5],
                                                                                       max_pool_size=2,
                                                                                       initializer=self.initializer)

        # Init V/D/J Masked Input Signal Encoding Layers
        self.v_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                             num_conv_batch_layers=6,
                                                                             kernel_size=[3,3,3,2,2,2,5],
                                                                             max_pool_size=2,
                                                                             conv_activation=tf.keras.layers.Activation(
                                                                                 'tanh'),
                                                                             initializer=self.initializer)

        self.d_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=64,
                                                                             num_conv_batch_layers=4,
                                                                             kernel_size=[3,3,3,3,5],
                                                                             max_pool_size=2,
                                                                             initializer=self.initializer)

        self.j_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=64,
                                                                             num_conv_batch_layers=4,
                                                                             kernel_size=[3,3,3,2,5],
                                                                             max_pool_size=2,
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
        self.intersection_loss_tracker = tf.keras.metrics.Mean(name="intersection_loss")
        # Track the segmentation loss
        self.total_segmentation_loss_tracker = tf.keras.metrics.Mean(name="segmentation_loss")
        # Track the classification loss
        self.classification_loss_tracker = tf.keras.metrics.Mean(name="classification_loss")
        # Track the mutation rate loss
        self.mutation_rate_loss_tracker = tf.keras.metrics.Mean(name="mutation_rate_loss")
        # Track the mutation indel count
        self.indel_count_loss_tracker = tf.keras.metrics.Mean(name="indel_count_loss")


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
            self.max_seq_length // 2, activation=act, name="mutation_rate_mid", kernel_initializer=self.initializer
        )
        self.mutation_rate_dropout = Dropout(0.05)
        self.mutation_rate_head = Dense(
            1, activation='relu', name="mutation_rate", kernel_initializer=self.initializer
            , kernel_constraint=MinMaxValueConstraint(0, 1)
        )

        self.indel_count_mid = Dense(
            self.max_seq_length // 2, activation=act, name="indel_count_mid", kernel_initializer=self.initializer
        )
        self.indel_count_dropout = Dropout(0.05)
        self.indel_count_head = Dense(
            1, activation='relu', name="indel_count", kernel_initializer=self.initializer
            , kernel_constraint=MinMaxValueConstraint(0, 50)
        )


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

        indel_count_mid = self.indel_count_mid(concatenated_signals)
        indel_count_mid = self.indel_count_dropout(indel_count_mid)
        indel_count = self.indel_count_head(indel_count_mid)

        return v_segment, d_segment, j_segment, mutation_rate,indel_count

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

        segmentation_features = self.segmentation_feature_extractor_block(input_embeddings)

        v_segment, d_segment, j_segment, mutation_rate, indel_count = self.predict_segments(segmentation_features)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)
        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)
        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, input_embeddings])
        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, input_embeddings])
        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, input_embeddings])

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self.v_feature_extraction_block(masked_sequence_v)
        d_feature_map = self.d_feature_extraction_block(masked_sequence_d)
        j_feature_map = self.j_feature_extraction_block(masked_sequence_j)

        # STEP 8: Predict The V,D and J genes
        v_allele, d_allele, j_allele = self._predict_vdj_set(v_feature_map, d_feature_map, j_feature_map)

        return {
            "v_segment": v_segment,
            "d_segment": d_segment,
            "j_segment": j_segment,
            "v_allele": v_allele,
            "d_allele": d_allele,
            "j_allele": j_allele,
            'mutation_rate': mutation_rate,
            'indel_count':indel_count
        }

    def get_segmentation_feature_map(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        segmentation_features = self.segmentation_feature_extractor_block(input_embeddings)
        return segmentation_features

    def get_v_latent_dimension(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        segmentation_features = self.segmentation_feature_extractor_block(input_embeddings)

        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(segmentation_features)

        reshape_masked_sequence_v = self.v_mask_reshape(v_segment)

        masked_sequence_v = self.v_mask_gate([reshape_masked_sequence_v, input_embeddings])

        v_feature_map = self.v_feature_extraction_block(masked_sequence_v)

        v_allele_latent = self.v_allele_mid(v_feature_map)

        return v_allele_latent

    def get_d_latent_dimension(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        segmentation_features = self.segmentation_feature_extractor_block(input_embeddings)

        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(segmentation_features)

        reshape_masked_sequence_j = self.j_mask_reshape(j_segment)

        masked_sequence_j = self.j_mask_gate([reshape_masked_sequence_j, input_embeddings])

        j_feature_map = self.j_feature_extraction_block(masked_sequence_j)

        j_allele_latent = self.j_allele_mid(j_feature_map)

        return j_allele_latent

    def get_j_latent_dimension(self, inputs):
        input_seq = self.reshape_and_cast_input(inputs["tokenized_sequence"])
        input_embeddings = self.input_embeddings(input_seq)

        segmentation_features = self.segmentation_feature_extractor_block(input_embeddings)

        v_segment, d_segment, j_segment, mutation_rate = self.predict_segments(segmentation_features)

        reshape_masked_sequence_d = self.d_mask_reshape(d_segment)

        masked_sequence_d = self.d_mask_gate([reshape_masked_sequence_d, input_embeddings])

        d_feature_map = self.d_feature_extraction_block(masked_sequence_d)

        d_allele_latent = self.d_allele_mid(d_feature_map)

        return d_allele_latent

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
                + mutation_rate_loss+indel_count_loss

        )

        return total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss,indel_count_loss

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            (
                total_loss, total_intersection_loss, total_segmentation_loss, classification_loss, mutation_rate_loss,indel_count_loss
            ) = self.multi_task_loss(y, y_pred)  # loss function

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        self.log_metrics(total_loss, total_intersection_loss, total_segmentation_loss, classification_loss,
                         mutation_rate_loss,indel_count_loss)

        metrics = self.get_metrics_log()

        return metrics

    def log_metrics(self, total_loss, total_intersection_loss, total_segmentation_loss, classification_loss,
                    mutation_rate_loss,indel_count_loss):
        # Compute our own metrics
        self.loss_tracker.update_state(total_loss)
        self.intersection_loss_tracker.update_state(total_intersection_loss)
        self.total_segmentation_loss_tracker.update_state(total_segmentation_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.mutation_rate_loss_tracker.update_state(mutation_rate_loss)
        self.indel_count_loss_tracker.update_state(indel_count_loss)

    def get_metrics_log(self):
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = self.loss_tracker.result()
        metrics["intersection_loss"] = self.intersection_loss_tracker.result()
        metrics["segmentation_loss"] = self.total_segmentation_loss_tracker.result()
        metrics["classification_loss"] = self.classification_loss_tracker.result()
        metrics["mutation_rate_loss"] = self.mutation_rate_loss_tracker.result()
        metrics["indel_count_loss"] = self.indel_count_loss_tracker.result()
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
