import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layers import Conv1D_and_BatchNorm, CutoutLayer, StartEndOutputNode, ProductivityHead, IndelCountHead, \
    MutationRateHead, AlleleClassificationHead, SegmentationConvResidualFeatureExtractionBlock
from .Layers import ClassificationConvResidualFeatureExtractionBlock, RegularizedConstrainedLogVar
from .Layers import TokenAndPositionEmbedding, MinMaxValueConstraint

class HeavyChainAlignAIR(nn.Module):
    """
    The AlignAIRR model for performing segmentation, mutation rate estimation,
    and allele classification tasks in heavy chain sequences.
    """
    def __init__(self, max_seq_length, v_allele_count, d_allele_count, j_allele_count):
        super(HeavyChainAlignAIR, self).__init__()

        # Model Parameters
        self.max_seq_length = max_seq_length
        self.v_allele_count = v_allele_count
        self.d_allele_count = d_allele_count
        self.j_allele_count = j_allele_count

        # Hyperparameters
        self.latent_size_factor = 2
        self.fblock_activation = nn.Tanh()
        self.classification_middle_layer_activation = nn.SiLU()
        self.initializer = nn.init.xavier_uniform_

        # Embedding layers
        self.input_embeddings = TokenAndPositionEmbedding(maxlen=self.max_seq_length, vocab_size=6, embed_dim=32)

        # Feature extraction blocks
        self.meta_feature_extractor_block = SegmentationConvResidualFeatureExtractionBlock(
            conv_activation=self.fblock_activation
        )
        self.v_segmentation_feature_block = SegmentationConvResidualFeatureExtractionBlock(
            conv_activation=self.fblock_activation
        )
        self.d_segmentation_feature_block = SegmentationConvResidualFeatureExtractionBlock(
            conv_activation=self.fblock_activation
        )
        self.j_segmentation_feature_block = SegmentationConvResidualFeatureExtractionBlock(
            conv_activation=self.fblock_activation
        )

        # classification feature blocks
        self.v_classification_feature_block = ClassificationConvResidualFeatureExtractionBlock(
            conv_activation=self.fblock_activation
        )
        self.d_classification_feature_block = ClassificationConvResidualFeatureExtractionBlock(
            conv_activation=self.fblock_activation
        )
        self.j_classification_feature_block = ClassificationConvResidualFeatureExtractionBlock(
            conv_activation=self.fblock_activation
        )

        # Mask layers
        self.v_mask_layer = CutoutLayer(max_size=self.max_seq_length, gene='V')
        self.d_mask_layer = CutoutLayer(max_size=self.max_seq_length, gene='D')
        self.j_mask_layer = CutoutLayer(max_size=self.max_seq_length, gene='J')

        # Classification layers
        self._init_classification_heads()

        # Segmentation prediction layers
        self._init_segmentation_predictions()

        # Initialize log variances for dynamic weighting
        self.setup_log_variances()

        self._init_masking_layers()

        self._init_meta_tasks()

    def _init_classification_heads(self):
        # V classification head
        self.v_allele_head = AlleleClassificationHead(576, latent_size_factor=self.latent_size_factor,
                                                      output_dim=self.v_allele_count)
        self.d_allele_head = AlleleClassificationHead(576, latent_size_factor=self.latent_size_factor,
                                                      output_dim=self.d_allele_count)
        self.j_allele_head = AlleleClassificationHead(576, latent_size_factor=self.latent_size_factor,
                                                      output_dim=self.j_allele_count)

    def setup_log_variances(self):
        """Initialize log variances for dynamic weighting."""
        self.log_var_v_start = RegularizedConstrainedLogVar()
        self.log_var_v_end = RegularizedConstrainedLogVar()
        self.log_var_d_start = RegularizedConstrainedLogVar()
        self.log_var_d_end = RegularizedConstrainedLogVar()
        self.log_var_j_start = RegularizedConstrainedLogVar()
        self.log_var_j_end = RegularizedConstrainedLogVar()
        self.log_var_v_allele_classification = RegularizedConstrainedLogVar()
        self.log_var_d_allele_classification = RegularizedConstrainedLogVar()
        self.log_var_j_allele_classification = RegularizedConstrainedLogVar()
        self.log_var_mutation = RegularizedConstrainedLogVar()
        self.log_var_indel = RegularizedConstrainedLogVar()
        self.log_var_productivity = RegularizedConstrainedLogVar()

    def _init_masking_layers(self):
        """
        Initialize masking layers for V, D, J.
        These layers will perform an element-wise product between the mask and embeddings.
        """

        def mask_gate_fn(embeddings, mask):
            """
            Applies the Hadamard product row-wise between embeddings and the mask.

            Args:
                embeddings (torch.Tensor): Shape (batch_size, embedding_dim, max_input_length)
                mask (torch.Tensor): Shape (batch_size, max_input_length)

            Returns:
                torch.Tensor: Masked embeddings of shape (batch_size, embedding_dim, max_input_length)
            """
            # Reshape mask to match the embeddings dimensions
            mask = mask.unsqueeze(1)  # Shape (batch_size, 1, max_input_length)
            return embeddings * mask  # Broadcasted element-wise multiplication

        # Initialize masking gates for V, D, J
        self.v_mask_gate = mask_gate_fn
        self.d_mask_gate = mask_gate_fn
        self.j_mask_gate = mask_gate_fn

    def _init_meta_tasks(self):
        self.mutation_rate_mid = nn.Linear(self.max_seq_length, self.max_seq_length)
        self.mutation_rate_dropout = nn.Dropout(0.05)
        self.mutation_rate_head = MutationRateHead(self.max_seq_length)

        self.indel_count_mid = nn.Linear(self.max_seq_length, self.max_seq_length)
        self.indel_count_dropout = nn.Dropout(0.05)
        self.indel_count_head = IndelCountHead(self.max_seq_length)

        self.productivity_feature_block = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=1)
        )
        self.productivity_flatten = nn.Flatten()

        self.productivity_dropout = nn.Dropout(0.05)
        self.productivity_head = ProductivityHead(self.max_seq_length)

    def _init_segmentation_predictions(self):
        act = F.gelu
        # Segmentation prediction layers
        self.v_start_out = StartEndOutputNode(576, 1)
        self.v_end_out = StartEndOutputNode(576, 1)

        self.d_start_out = StartEndOutputNode(576, 1)
        self.d_end_out = StartEndOutputNode(576, 1)

        self.j_start_out = StartEndOutputNode(576, 1)
        self.j_end_out = StartEndOutputNode(576, 1)

    def forward(self, inputs):
        """
        Forward pass for the model.
        """
        # Step 1: Input embeddings
        input_embeddings = self.input_embeddings(inputs)

        # Step 2: Feature extraction
        meta_features = self.meta_feature_extractor_block(input_embeddings)
        v_segment_features = self.v_segmentation_feature_block(input_embeddings)
        d_segment_features = self.d_segmentation_feature_block(input_embeddings)
        j_segment_features = self.j_segmentation_feature_block(input_embeddings)

        # Step 3: Segmentation predictions
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

        # Step 4: Classification predictions
        v_feature_mask = self.v_mask_layer([v_start, v_end])
        d_feature_mask = self.d_mask_layer([d_start, d_end])
        j_feature_mask = self.j_mask_layer([j_start, j_end])

        masked_sequence_v = self.v_mask_gate(input_embeddings, v_feature_mask)
        masked_sequence_d = self.d_mask_gate(input_embeddings, d_feature_mask)
        masked_sequence_j = self.j_mask_gate(input_embeddings, j_feature_mask)

        v_features = self.v_classification_feature_block(masked_sequence_v)
        d_features = self.d_classification_feature_block(masked_sequence_d)
        j_features = self.j_classification_feature_block(masked_sequence_j)

        v_allele = self.v_allele_head(v_features)
        d_allele = self.d_allele_head(d_features)
        j_allele = self.j_allele_head(j_features)

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
