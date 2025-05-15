import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from GenAIRR.data import builtin_tcrb_data_config
from scipy.stats import entropy
from tensorflow.keras.callbacks import ReduceLROnPlateau

from AlignAIR.Data import HeavyChainDataset
from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
from AlignAIR.Trainers import Trainer
from AlignAIR.Utilities.step_utilities import DataConfigLibrary

sns.set_context('poster')

class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, dataconfig_lib,datasetobject,sample_data,update_every=100, output_dir="web_output"):
        super().__init__()
        self.dataconfig_lib=dataconfig_lib
        self.datasetobject = datasetobject
        self.update_every = update_every
        self.sample_data = sample_data.iloc[:50,:]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.latest_prediction = None

    def on_batch_end(self, batch, logs=None):
        if batch % self.update_every == 0:
            inp = {'tokenized_sequence':self.datasetobject.encode_and_pad_sequences(self.sample_data['sequence'])[0]}
            sample_prediction = self.model.predict(inp)
            self.sample_prediction=sample_prediction
            self._generate_analysis(batch)

    def assess_mutation_rate(self):
        true_mutation_rate = self.sample_data['mutation_rate']
        predicted_mutation_rate = self.model.predict(self.sample_data)

    def generate_allele_likelihood_distribution(self,step):
        v_allele = self.sample_prediction['v_allele'][:5]
        d_allele = self.sample_prediction['d_allele'][:5]
        j_allele = self.sample_prediction['j_allele'][:5]
        groundtruth_v = self.sample_data['v_call'][:5]
        groundtruth_d = self.sample_data['d_call'][:5]
        groundtruth_j = self.sample_data['j_call'][:5]

        v_alleles = []
        v_names = list(sorted(self.dataconfig_lib.get_allele_dict('v').keys()))
        for i in v_allele:
            v_alleles.append({name:j for name,j in zip(v_names, i)})
        d_alleles = []
        d_names = list(sorted(self.dataconfig_lib.get_allele_dict('d').keys()))
        for i in d_allele:
            d_alleles.append({name:j for name,j in zip(d_names, i)})
        j_alleles = []
        j_names = list(sorted(self.dataconfig_lib.get_allele_dict('j').keys()))
        for i in j_allele:
            j_alleles.append({name:j for name,j in zip(j_names, i)})

        gene_results = {
            'v': {
                'predictions': v_alleles,
                'groundtruth': groundtruth_v
            },
            'd': {
                'predictions': d_alleles,
                'groundtruth': groundtruth_d
            },
            'j': {
                'predictions': j_alleles,
                'groundtruth': groundtruth_j
            }
        }

        for gene in gene_results:
            plt.figure(figsize=(30,20))
            plt.title(f"Allele likelihood distribution for {gene} genes")
            allele_likelihoods = gene_results[gene]['predictions']
            allele_names = list(allele_likelihoods[0].keys())
            true_alleles = gene_results[gene]['groundtruth']

            for i, allele in enumerate(allele_likelihoods):
                # Sort the allele dictionary
                sorted_allele = {k: v for k, v in sorted(allele.items(), key=lambda item: item[0])}
                allele_names_sorted = list(sorted_allele.keys())
                allele_values_sorted = list(sorted_allele.values())

                plt.subplot(5, 1, i + 1)
                _entropy = entropy(allele_values_sorted)
                plt.title(f"Sample {i + 1} Entropy: {_entropy:.2f}")
                plt.bar(allele_names_sorted, allele_values_sorted, color='blue', alpha=0.7)

                # Mark the true allele
                true_allele = true_alleles[i]
                if true_allele in sorted_allele:
                    true_allele_index = allele_names_sorted.index(true_allele)
                    plt.bar(allele_names_sorted[true_allele_index],
                            allele_values_sorted[true_allele_index],
                            color='red', alpha=0.7)

                plt.xticks([])
                plt.xlabel('Allele')
                plt.ylabel('Likelihood')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"allele_likelihood_{gene}_{step}.png"),
                        bbox_inches='tight')





    def _generate_analysis(self, step):
        #predictions = self.model.predict(sample_data)

        latent_anchor_plot_paths = self.project_genes_to_latent_space(step)
        self.generate_allele_likelihood_distribution(step)
        #mutation_rate_plot = self.assess_mutation_rate()


    def project_genes_to_latent_space(self, step):
        genes = ['v', 'd', 'j']
        plots_saved = []
        latent_anchors = {}
        for gene in genes:
            alleles = self.dataconfig_lib.get_allele_dict(gene)
            alleles = {k:v for k,v in sorted(alleles.items(), key=lambda item: item[0])} # sort keys
            # sort keys
            sequences = list(alleles.values())
            encoded_sequences = self.datasetobject.encode_and_pad_sequences(sequences)[0]
            encoded_sequences = {'tokenized_sequence':encoded_sequences}
            if gene == 'v':
                preds = self.model.get_v_latent_dimension(encoded_sequences)
            elif gene == 'd':
                preds = self.model.get_d_latent_dimension(encoded_sequences)
            elif gene == 'j':
                preds = self.model.get_j_latent_dimension(encoded_sequences)
            latent_anchors[gene] = (preds,list(alleles))


        for gene in genes:
            plt.figure(figsize=(20,10))
            plt.title(f"Latent space projection of {gene} genes")
            categories = [i.split('-')[0].split('*')[0] for i in latent_anchors[gene][1]]
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(latent_anchors[gene][0])

            # Create a color map for the categories
            unique_categories = list(set(categories))
            # use tab20
            colors = plt.get_cmap('tab20', len(unique_categories))

            color_map = {category: colors(i) for i, category in enumerate(unique_categories)}
            # Map the categories to colors
            colors = [color_map[category] for category in categories]


            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            # add legend with colors outside the plot
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=category,
                                  markerfacecolor=color_map[category], markersize=10) for category in unique_categories]
            plt.legend(handles=handles, title='Allele', bbox_to_anchor=(1.05, 1), loc='upper left')


            plt.savefig(os.path.join(self.output_dir, f"latent_space_{gene}_{step}.png"),
                        bbox_inches='tight')

            plots_saved.append(os.path.join(self.output_dir, f"latent_space_{gene}_{step}.png"))

        return plots_saved



import numpy as np
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

from AlignAIR.Models.Layers import Conv1D_and_BatchNorm, CutoutLayer, AverageLastLabel, EntropyMetric
from AlignAIR.Models.Layers import ConvResidualFeatureExtractionBlock, RegularizedConstrainedLogVar
from AlignAIR.Models.Layers import (
    TokenAndPositionEmbedding, MinMaxValueConstraint
)


class AverageMaskSize(tf.keras.metrics.Mean):
    def __init__(self, gene: str, **kwargs):
        """
        Tracks the average size (number of positions = 1) in the segmentation mask for a given gene.

        Args:
            gene (str): One of "v", "d", "j".
        """
        super().__init__(name=f"{gene}_mask_avg_size", **kwargs)
        self.gene = gene
        self.mask_key = f"{gene}_mask"

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Expects y_pred to contain a key like "v_mask", "d_mask", "j_mask"
        corresponding to a tensor of shape (B, L, 1).
        """
        mask = tf.cast(y_pred[self.mask_key], tf.float32)
        sizes = tf.reduce_sum(mask, axis=[1, 2])  # shape: (B,)
        return super().update_state(sizes, sample_weight)


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

    def __init__(self, max_seq_length, v_allele_count, d_allele_count, j_allele_count, v_allele_latent_size=None,
                 d_allele_latent_size=None, j_allele_latent_size=None):
        super(HeavyChainAlignAIRR, self).__init__()

        # weight initialization distribution
        self.initializer = tf.keras.initializers.GlorotUniform()  # RandomNormal(mean=0.1, stddev=0.02)

        # Model Params
        self.max_seq_length = int(max_seq_length)
        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count
        self.v_allele_latent_size = v_allele_latent_size
        self.d_allele_latent_size = d_allele_latent_size
        self.j_allele_latent_size = j_allele_latent_size

        # Hyperparams + Constants
        self.classification_keys = ["v_allele", "d_allele", "j_allele"]
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = "swish"
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.segmentation_weight, self.classification_weight, self.intersection_weight = (0.5, 0.5, 0.5)

        self.BCE_SMOOTH = tf.keras.losses.BinaryCrossentropy()

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
        self.d_segmentation_feature_block = ConvResidualFeatureExtractionBlock(filter_size=128,
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
        self.d_mask_layer = CutoutLayer(gene='D', max_size=self.max_seq_length)
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

        self.d_feature_extraction_block = ConvResidualFeatureExtractionBlock(filter_size=128,
                                                                             num_conv_batch_layers=4,
                                                                             kernel_size=[3, 3, 2, 2, 5],
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
        # =========== D HEADS ======================
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        self._init_j_classification_layers()

    def setup_log_variances(self):
        """Initialize log variances for dynamic weighting."""
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

        self.v_mask_size_metric = AverageMaskSize("v")
        self.d_mask_size_metric = AverageMaskSize("d")
        self.j_mask_size_metric = AverageMaskSize("j")

        # Add custom metrics for monitoring
        self.average_last_label_tracker = AverageLastLabel(name="average_last_label")
        self.v_allele_entropy_tracker = EntropyMetric(allele_name="v_allele")
        self.d_allele_entropy_tracker = EntropyMetric(allele_name="d_allele")
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
        self.d_mask_gate = Multiply()
        self.d_mask_reshape = Reshape((self.max_seq_length, 1))
        self.j_mask_gate = Multiply()
        self.j_mask_reshape = Reshape((self.max_seq_length, 1))

    def _init_v_classification_layers(self):
        self.v_allele_mid = Dense(
            self.v_allele_count * self.latent_size_factor if self.v_allele_latent_size is None else self.v_allele_latent_size,
            activation=self.classification_middle_layer_activation,
            name="v_allele_middle", kernel_initializer=self.initializer,
        )

        self.v_allele_call_head = Dense(self.v_allele_count, activation="sigmoid", name="v_allele")

    def _init_d_classification_layers(self):
        self.d_allele_mid = Dense(
            self.d_allele_count * self.latent_size_factor if self.d_allele_latent_size is None else self.d_allele_latent_size,
            kernel_regularizer=regularizers.l2(0.01),
            activation=self.classification_middle_layer_activation,
            name="d_allele_middle",
        )

        self.d_allele_call_head = Dense(
            self.d_allele_count, activation="sigmoid", name="d_allele", kernel_regularizer=regularizers.l2(0.05)
        )

    def _init_j_classification_layers(self):
        self.j_allele_mid = Dense(
            self.j_allele_count * self.latent_size_factor if self.j_allele_latent_size is None else self.j_allele_latent_size,
            activation=self.classification_middle_layer_activation,
            name="j_allele_middle",
        )

        self.j_allele_call_head = Dense(
            self.j_allele_count, activation="sigmoid", name="j_allele", kernel_regularizer=regularizers.l1(0.01)
        )

    def _init_segmentation_predictions(self):
        # act = tf.keras.layers.LeakyReLU()
        # def gelu_custom(x):
        #     """
        #     Custom implementation of GELU activation.
        #     """
        #     pi = tf.constant(3.141592653589793, dtype=x.dtype)  # Define π
        #     cdf = 0.5 * (1.0 + tf.tanh((tf.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x, 3)))))
        #     return x * cdf
        # act = gelu_custom


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
        d_segment_features = self.d_segmentation_feature_block(input_embeddings)

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

        # reshape_masked_sequence_v = tf.expand_dims(reshape_masked_sequence_v, -1)
        # reshape_masked_sequence_d = tf.expand_dims(reshape_masked_sequence_d, -1)
        # reshape_masked_sequence_j = tf.expand_dims(reshape_masked_sequence_j, -1)
        reshape_masked_sequence_v = self.v_mask_reshape(reshape_masked_sequence_v)
        reshape_masked_sequence_d = self.d_mask_reshape(reshape_masked_sequence_d)
        reshape_masked_sequence_j = self.j_mask_reshape(reshape_masked_sequence_j)

        # import pdb
        # pdb.set_trace()

        self.latest_masks = {
            "v_mask": reshape_masked_sequence_v,
            "d_mask": reshape_masked_sequence_d,
            "j_mask": reshape_masked_sequence_j
        }

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

    def hierarchical_loss(self, y_true, y_pred):
        # Extract the segmentation and classification outputs

        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]

        # Short-D Scaling Factor
        short_d_prob = classification_pred[1][:, -1]  # Probabilities for the bad input
        short_d_scaling_factor = 1.0 - short_d_prob
        d_length_pred = y_pred['d_end'] - y_pred['d_start']

        # Segmentation Loss
        v_start_loss = tf.keras.losses.mean_absolute_error(y_true['v_start'], y_pred['v_start'])
        v_end_loss = tf.keras.losses.mean_absolute_error(y_true['v_end'], y_pred['v_end'])

        d_start_loss = tf.keras.losses.mean_absolute_error(y_true['d_start'], y_pred['d_start'])
        d_end_loss = tf.keras.losses.mean_absolute_error(y_true['d_end'], y_pred['d_end'])

        j_start_loss = tf.keras.losses.mean_absolute_error(y_true['j_start'], y_pred['j_start'])
        j_end_loss = tf.keras.losses.mean_absolute_error(y_true['j_end'], y_pred['j_end'])

        # Calculate the precision as the inverse of the exponential of each task's log variance
        weighted_v_start = v_start_loss * self.log_var_v_start(v_start_loss)
        weighted_v_end = v_end_loss * self.log_var_v_end(v_end_loss)

        weighted_d_start = d_start_loss * self.log_var_d_start(d_start_loss)
        weighted_d_end = d_end_loss * self.log_var_d_end(d_end_loss)

        weighted_j_start = j_start_loss * self.log_var_j_start(j_start_loss)
        weighted_j_end = j_end_loss * self.log_var_j_end(j_end_loss)

        ##########################################################################################################

        segmentation_loss = weighted_v_start + weighted_v_end + weighted_d_start + weighted_d_end + weighted_j_start + weighted_j_end

        # Classification Loss

        clf_v_loss = tf.keras.losses.binary_crossentropy(classification_true[0], classification_pred[0])
        # D loss
        penalty_factor = 1.0
        last_label_penalty_factor = 1.0
        # Binary crossentropy
        bce = self.BCE_SMOOTH(classification_true[1], classification_pred[1])
        clf_d_loss = bce  # (bce + last_label_penalty)

        clf_j_loss = tf.keras.losses.binary_crossentropy(classification_true[2], classification_pred[2])

        precision_v_classification = 1#self.log_var_v_classification(clf_v_loss)
        precision_d_classification = 1#self.log_var_d_classification(clf_d_loss)
        precision_j_classification = 1#self.log_var_j_classification(clf_j_loss)
        classification_loss = precision_v_classification * clf_v_loss + precision_d_classification * clf_d_loss + precision_j_classification * clf_j_loss

        # Custom penalty for short D lengths with high short D likelihood
        threshold = 5
        penalty_factor = 1.0
        short_d_length_penalty = tf.reduce_mean(
            penalty_factor * tf.cast(d_length_pred < threshold, tf.float32) * short_d_prob
        )
        classification_loss += short_d_length_penalty

        # Mutation Loss
        mutation_rate_loss = tf.keras.losses.mean_absolute_error(y_true['mutation_rate'], y_pred['mutation_rate'])
        # Indel Count Loss
        indel_count_loss = tf.keras.losses.mean_absolute_error(y_true['indel_count'], y_pred['indel_count'])

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

        y_pred['v_mask'] = self.latest_masks['v_mask']
        y_pred['d_mask'] = self.latest_masks['d_mask']
        y_pred['j_mask'] = self.latest_masks['j_mask']

        self.average_last_label_tracker.update_state(y, y_pred)
        self.v_allele_entropy_tracker.update_state(y, y_pred)
        self.d_allele_entropy_tracker.update_state(y, y_pred)
        self.j_allele_entropy_tracker.update_state(y, y_pred)

        self.v_mask_size_metric.update_state(y, y_pred)
        self.d_mask_size_metric.update_state(y, y_pred)
        self.j_mask_size_metric.update_state(y, y_pred)

        metrics = self.get_metrics_log()

        metrics['v_mask_avg_size'] = self.v_mask_size_metric.result()
        metrics['d_mask_avg_size'] = self.d_mask_size_metric.result()
        metrics['j_mask_avg_size'] = self.j_mask_size_metric.result()

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
        metrics['average_last_label'] = self.average_last_label_tracker.result()
        metrics['v_allele_entropy'] = self.v_allele_entropy_tracker.result()
        metrics['d_allele_entropy'] = self.d_allele_entropy_tracker.result()
        metrics['j_allele_entropy'] = self.j_allele_entropy_tracker.result()
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
import tensorflow.keras.backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



seed = 42
np.random.seed(seed)  # Set the random seed
tf.random.set_seed(seed)  # Set the random seed
random.seed(seed)  # Set the random seed

patience = 5
epoch_to_change = 2

### Start - CallBacks Definitions ###

reduce_lr = ReduceLROnPlateau(
    monitor="loss",
    factor=0.9,
    patience=10,
    min_delta=0.01,
    mode="auto",
)



# Define other parameters
epochs = 1200
batch_size = 512
noise_type = (
    "S5F_rate"  # Can be of types: ["S5F_rate", "S5F_20", "s5f_opposite", "uniform"]
)



TRAIN_DATASET = r"C:\Users\tomas\Desktop\AlignAIRR\tests\TCRB_Sample_K2_Data.csv"
chain_type = 'tcrb'


train_dataset = HeavyChainDataset(data_path=TRAIN_DATASET,
                                dataconfig=builtin_tcrb_data_config(),
                                batch_size=batch_size,
                                max_sequence_length=576,
                                batch_read_file=True)





print('Starting The Training...')
print('Train Dataset Path: ',TRAIN_DATASET)

import tensorflow.keras.backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def hamming_loss_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return K.mean(K.not_equal(y_true, K.round(y_pred)))
def subset_accuracy_m(y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(K.round(y_pred), 'int32')
    return K.mean(K.all(K.equal(y_true, y_pred), axis=-1))


# trainer = Trainer(
#     model= HeavyChainAlignAIRR(**train_dataset.generate_model_params()),
#     batch_size=512,
#     epochs=epochs,
#     steps_per_epoch=150_000,
#     verbose=1,
#     classification_metric=[[tf.keras.metrics.AUC(),f1,hamming_loss_m,subset_accuracy_m],
#                         [tf.keras.metrics.AUC(),f1,hamming_loss_m,subset_accuracy_m],
#                         [tf.keras.metrics.AUC(),f1,hamming_loss_m,subset_accuracy_m]],
#     regression_metric=tf.keras.losses.mean_absolute_error,
#     callbacks=[
#         reduce_lr,
#         #p1p11_evaluation_callback,
#     ]
# )
#
# trainer.model.compile(
#     optimizer=tf.keras.optimizers.Adam(clipnorm=1),
#     loss=None,
#     metrics={
#         'v_allele': [f1, tf.keras.metrics.AUC(name="auc_v")],
#         'd_allele': [f1, tf.keras.metrics.AUC(name="auc_d")],
#         'j_allele': [f1, tf.keras.metrics.AUC(name="auc_j")],
#     }
# )
#
# # Train the model
# trainer.train(train_dataset)



if __name__ == "__main__":
    from GenAIRR.data import builtin_tcrb_data_config

    data_path = r'C:\Users\tomas\Desktop\AlignAIRR\tests\edf_sample.csv'
    sample_data  = pd.read_csv(data_path)
    train_dataset = HeavyChainDataset(data_path=data_path,
                                      dataconfig=builtin_tcrb_data_config(), batch_read_file=True,
                                      max_sequence_length=576)

    n_v_alleles = (train_dataset.v_allele_count)

    model_parmas = train_dataset.generate_model_params()
    #model_parmas['v_allele_latent_size'] = n_v_alleles*3
    model = HeavyChainAlignAIRR(**model_parmas)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1),
        loss=None,
        metrics = {
            'v_allele': [f1, tf.keras.metrics.AUC(name="auc_v")],
            'd_allele': [f1, tf.keras.metrics.AUC(name="auc_d")],
            'j_allele': [f1, tf.keras.metrics.AUC(name="auc_j")],
        }
    )

    dataconfig_lib = DataConfigLibrary()
    dataconfig_lib.mount_type('tcrb')

    monitor = TrainingMonitor(update_every=10,
                              datasetobject=train_dataset,
                              sample_data = sample_data,
                              output_dir="C:/Users/tomas/Downloads/AlignAIR/web_output",
                              dataconfig_lib=dataconfig_lib)

    trainer = Trainer(
        model=model,
        epochs=10,
        batch_size=512,
        steps_per_epoch=150_000,
        verbose=1,
        classification_metric={
            'v_allele': [f1, tf.keras.metrics.AUC(name="auc_v")],
            'd_allele': [f1, tf.keras.metrics.AUC(name="auc_d")],
            'j_allele': [f1, tf.keras.metrics.AUC(name="auc_j")],
        },
        regression_metric=tf.keras.losses.mae,
        callbacks=[monitor]
    )


    # Train the model
    trainer.train(train_dataset)


    with open('C:/Users/tomas/Downloads/AlignAIR/history.pkl','wb') as h:
        import pickle
        pickle.dump(pd.DataFrame(trainer.history.history),h)




#
# if __name__ == "__main__":
#     from AlignAIR.Utilities.step_utilities import DataConfigLibrary, FileInfo
#     from AlignAIR.Data.PredictionDataset import PredictionDataset
#     from AlignAIR.Models.HeavyChain import HeavyChainAlignAIRR
#     # Example usage
#     # Load your trained model
#     dataset = PredictionDataset(576)
#     dataconfig_lib = DataConfigLibrary()
#     dataconfig_lib.mount_type('heavy')
#
#     model_params = {
#         "max_seq_length": 576,
#         "v_allele_count": len(dataconfig_lib.get_allele_dict('v')),
#         "d_allele_count": len(dataconfig_lib.get_allele_dict('d')) + 1,
#         "j_allele_count": len(dataconfig_lib.get_allele_dict('j')),
#     }
#
#     model = HeavyChainAlignAIRR(**model_params)
#
#     model.build({'tokenized_sequence': (576, 1)})
#     model.load_weights(
#         "C:/Users/tomas/Desktop/AlignAIRR/tests/AlignAIRR_S5F_OGRDB_V8_S5F_576_Balanced_V2")
#
#     sample_data = pd.read_csv('C:/Users/tomas/Downloads/IGH_Sample_Data.csv')
#     sample_data['c_call'] = None
#     sample_data = sample_data[sample_data['sequence'].notna()]
#
#     # Create the callback instance
#     monitor = TrainingMonitor(update_every=100,
#                               datasetobject=dataset,
#                               sample_data = sample_data,
#                               output_dir="C:/Users/tomas/Downloads/AlignAIR/web_output",
#                               dataconfig_lib=dataconfig_lib)
#
#     # Inject model manually
#     monitor.model = model
#
#     sample_prediction = monitor.model.predict(monitor.datasetobject.process_sequences(monitor.sample_data['sequence'].to_list()),batch_size=64)
#     monitor.sample_prediction = sample_prediction
#
#     # Manually call the analysis method
#     monitor._generate_analysis(step=0)