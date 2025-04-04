import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D_and_BatchNorm(nn.Module):
    def __init__(self, in_channels=32, filters=16, kernel=3, max_pool=2, activation=None):
        super(Conv1D_and_BatchNorm, self).__init__()

        # Define the Conv1D layers
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=filters,
                               kernel_size=kernel,
                               padding='same',  # Achieved manually in PyTorch
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=filters,
                               out_channels=filters,
                               kernel_size=kernel,
                               padding='same',
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=filters,
                               out_channels=filters,
                               kernel_size=kernel,
                               padding='same',
                               bias=True)

        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(filters, eps=0.8, momentum=0.1)

        # Activation layer
        self.activation = activation if activation else nn.LeakyReLU()

        # MaxPooling layer
        self.max_pool = nn.MaxPool1d(kernel_size=max_pool)

    def forward(self, x):
        # Apply Conv1D layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Apply Batch Normalization
        x = self.batch_norm(x)

        # Apply Activation
        x = self.activation(x)

        # Apply MaxPooling
        x = self.max_pool(x)

        return x


class CutoutLayer(nn.Module):
    def __init__(self, max_size, gene):
        """
        A PyTorch implementation of the CutoutLayer.

        Args:
            max_size (int): The maximum size of the binary mask.
            gene (str): The type of gene ('V', 'D', or 'J') to determine specific behavior.
        """
        super(CutoutLayer, self).__init__()
        self.max_size = max_size
        self.gene = gene

    def round_output(self, dense_output):
        """
        Rounds the output values to the range [0, max_size] and casts to float.
        """
        max_value, _ = torch.max(dense_output, dim=-1, keepdim=True)
        max_value = torch.clamp(max_value, min=0, max=self.max_size)
        return max_value.float()

    def _create_mask(self, dense_start, dense_end, batch_size):
        """
        Creates a binary mask based on start and end predictions.
        Includes the end position in the mask.
        """
        x = self.round_output(dense_start)
        y = self.round_output(dense_end)

        # Create a range tensor [0, max_size)
        indices = torch.arange(0, self.max_size, device=dense_start.device).float().view(1, -1)
        indices = indices.expand(batch_size, -1)  # Shape: (batch_size, max_size)

        # Generate the binary mask (inclusive of the end position)
        R = (indices >= x) & (indices <= y)
        R = R.float()
        return R

    def forward(self, inputs):
        """
        Forward pass for the CutoutLayer.

        Args:
            inputs (tuple): A tuple containing dense_start and dense_end tensors.

        Returns:
            torch.Tensor: A binary mask with shape (batch_size, max_size).
        """
        dense_start, dense_end = inputs
        batch_size = dense_start.size(0)

        if self.gene in {'V', 'D', 'J'}:
            return self._create_mask(dense_start, dense_end, batch_size)
        else:
            raise ValueError(f"Unsupported gene type: {self.gene}")
class CutoutLayerV2(nn.Module):
    def __init__(self, max_size, gene):
        """
        A PyTorch implementation of the CutoutLayer.

        Args:
            max_size (int): The maximum size of the binary mask.
            gene (str): The type of gene ('V', 'D', or 'J') to determine specific behavior.
        """
        super(CutoutLayer, self).__init__()
        self.max_size = max_size
        self.gene = gene

    def _create_mask(self, dense_start, dense_end, batch_size):
        """
        Creates a binary mask based on start and end predictions.
        Rounds the start and end positions to the nearest integers.

        Args:
            dense_start (torch.Tensor): Predicted start indices (batch_size, 1).
            dense_end (torch.Tensor): Predicted end indices (batch_size, 1).
            batch_size (int): The batch size.

        Returns:
            torch.Tensor: A binary mask with shape (batch_size, max_size).
        """
        # Round start and end to the nearest integers
        start = torch.round(dense_start).long()
        end = torch.round(dense_end).long()

        # Clamp start and end to ensure they are within valid range
        start = torch.clamp(start, min=0, max=self.max_size - 1)
        end = torch.clamp(end, min=0, max=self.max_size - 1)

        # Create a range tensor [0, max_size)
        indices = torch.arange(0, self.max_size, device=dense_start.device).view(1, -1)
        indices = indices.expand(batch_size, -1)  # Shape: (batch_size, max_size)

        # Generate the binary mask (inclusive of the end position)
        mask = (indices >= start) & (indices <= end)
        return mask.float()

    def forward(self, inputs):
        """
        Forward pass for the CutoutLayer.

        Args:
            inputs (tuple): A tuple containing dense_start and dense_end tensors.

        Returns:
            torch.Tensor: A binary mask with shape (batch_size, max_size).
        """
        dense_start, dense_end = inputs
        batch_size = dense_start.size(0)

        if self.gene in {'V', 'D', 'J'}:
            return self._create_mask(dense_start, dense_end, batch_size)
        else:
            raise ValueError(f"Unsupported gene type: {self.gene}")
class SegmentationConvResidualFeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels=32, filters=128, out_shape=576, conv_activation=None):
        super(SegmentationConvResidualFeatureExtractionBlock, self).__init__()

        # Activation function
        self.conv_activation = conv_activation if conv_activation else nn.LeakyReLU()

        # Define sequential Conv1D_and_BatchNorm layers
        self.convbatch_1 = Conv1D_and_BatchNorm(
            in_channels=in_channels, filters=filters, kernel=3, max_pool=2, activation=self.conv_activation)
        self.convbatch_2 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=3, max_pool=2, activation=self.conv_activation)
        self.convbatch_3 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=3, max_pool=2, activation=self.conv_activation)
        self.convbatch_4 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=2, max_pool=2, activation=self.conv_activation)
        self.convbatch_5 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=5, max_pool=2, activation=self.conv_activation)

        # Residual channel for skip connection
        # self.residual_channel = nn.Conv1d(
        #     in_channels=in_channels, out_channels=filters, kernel_size=1, padding="same", bias=True)

        # Residual connection layers for each stage
        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=32 if i == 0 else filters,  # Match input channels dynamically
                          out_channels=filters, kernel_size=1, stride=2, bias=True),
                nn.BatchNorm1d(filters)
            ) for i in range(5)  # Residual updates for convbatch_2 to convbatch_5
        ])

        self.dense_reshaper = nn.Linear(2304, out_shape)
        self.feature_flatten = nn.Flatten()

    def forward(self, x):
        # Initial residual (no change needed for the first layer)
        residual = self.residual_layers[0](x)
        x = self.convbatch_1(x)  # First convolution and max-pooling step

        x = x + residual  # Add the residual connection

        # Apply sequential layers with residual connections, updating the residual path
        for i, convbatch in enumerate([self.convbatch_2, self.convbatch_3, self.convbatch_4, self.convbatch_5]):
            # Update the residual to match the reduced dimensions
            residual = self.residual_layers[i + 1](residual)
            x = convbatch(x) + residual  # Add the residual connection

        x = self.feature_flatten(x)
        x = self.dense_reshaper(x)
        return x

class ClassificationConvResidualFeatureExtractionBlock(nn.Module):
    def __init__(self, in_channels=32, filters=128, out_shape=576, conv_activation=None):
        super(ClassificationConvResidualFeatureExtractionBlock, self).__init__()

        # Activation function
        self.conv_activation = conv_activation if conv_activation else nn.LeakyReLU()

        # Define sequential Conv1D_and_BatchNorm layers
        self.convbatch_1 = Conv1D_and_BatchNorm(
            in_channels=in_channels, filters=filters, kernel=3, max_pool=2, activation=self.conv_activation)
        self.convbatch_2 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=3, max_pool=2, activation=self.conv_activation)
        self.convbatch_3 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=3, max_pool=2, activation=self.conv_activation)
        self.convbatch_4 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=2, max_pool=2, activation=self.conv_activation)
        self.convbatch_5 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=2, max_pool=2, activation=self.conv_activation)
        self.convbatch_6 = Conv1D_and_BatchNorm(
            in_channels=filters, filters=filters, kernel=2, max_pool=2, activation=self.conv_activation)

        # Residual channel for skip connection
        # self.residual_channel = nn.Conv1d(
        #     in_channels=in_channels, out_channels=filters, kernel_size=1, padding="same", bias=True)

        # Residual connection layers for each stage
        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=32 if i == 0 else filters,  # Match input channels dynamically
                          out_channels=filters, kernel_size=1, stride=2, bias=True),
                nn.BatchNorm1d(filters)
            ) for i in range(7)  # Residual updates for convbatch_2 to convbatch_5
        ])

        self.dense_reshaper = nn.Linear(1152, out_shape)
        self.feature_flatten = nn.Flatten()

    def forward(self, x):
        # Initial residual (no change needed for the first layer)
        residual = self.residual_layers[0](x)
        x = self.convbatch_1(x)  # First convolution and max-pooling step

        x = x + residual  # Add the residual connection

        # Apply sequential layers with residual connections, updating the residual path
        for i, convbatch in enumerate([self.convbatch_2, self.convbatch_3, self.convbatch_4, self.convbatch_5,
                                       self.convbatch_6]):
            # Update the residual to match the reduced dimensions
            residual = self.residual_layers[i + 1](residual)
            x = convbatch(x) + residual  # Add the residual connection

        x = self.feature_flatten(x)
        x = self.dense_reshaper(x)
        return x


class RegularizedConstrainedLogVar(nn.Module):
    def __init__(self, initial_value=1.0, min_log_var=-3, max_log_var=1, regularizer_weight=0.01):
        """
        A PyTorch implementation of RegularizedConstrainedLogVar.

        Parameters:
        - initial_value: Initial value for log(var).
        - min_log_var: Minimum value for log(var).
        - max_log_var: Maximum value for log(var).
        - regularizer_weight: Weight for the regularization term.
        """
        super(RegularizedConstrainedLogVar, self).__init__()
        self.log_var = nn.Parameter(
            torch.log(torch.tensor(initial_value, dtype=torch.float32))
        )
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.regularizer_weight = regularizer_weight

    def forward(self):
        """
        Forward pass to compute the regularized constrained log variance.

        Returns:
        - precision: exp(-clipped_log_var), used for weighting losses.
        - regularization_loss: Regularization penalty to be added to total loss.
        """
        # Apply constraints using torch.clamp
        clipped_log_var = torch.clamp(self.log_var, self.min_log_var, self.max_log_var)

        # Compute regularization loss
        regularization_loss = self.regularizer_weight * F.relu(-clipped_log_var - 2)

        # Return precision (exp(-log(var))) and the regularization loss
        precision = torch.exp(-clipped_log_var)
        return precision, regularization_loss


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos_emb = nn.Embedding(num_embeddings=maxlen, embedding_dim=embed_dim)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        token_embeddings = self.token_emb(x)  # Shape: (batch_size, seq_len, embed_dim)
        position_embeddings = self.pos_emb(positions)  # Shape: (batch_size, seq_len, embed_dim)
        # Transpose for Conv1D (batch_size, embed_dim, seq_len)
        return (token_embeddings + position_embeddings).permute(0, 2, 1)


class MinMaxValueConstraint:
    def __init__(self, min_value, max_value):
        """
        Constraint that clips values (not weights) to be within a specified range.

        Parameters:
        - min_value: Minimum allowed value.
        - max_value: Maximum allowed value.
        """
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x):
        """
        Applies the constraint by clipping the values.

        Parameters:
        - x: The tensor to constrain.

        Returns:
        - Clipped tensor.
        """
        return torch.clamp(x, self.min_value, self.max_value)

    def get_config(self):
        """
        Returns the configuration of the constraint.

        Returns:
        - Dictionary containing min and max values.
        """
        return {'min_value': self.min_value, 'max_value': self.max_value}

class MinMaxWeightConstraint:
    def __init__(self, min_value, max_value):
        """
        Constraint that clips weights to be within a specified range.

        Parameters:
        - min_value: Minimum allowed weight value.
        - max_value: Maximum allowed weight value.
        """
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, module):
        """
        Applies the constraint by clipping the weights of the given module.

        Parameters:
        - module: The PyTorch module to constrain (e.g., nn.Linear or nn.Conv2d).

        Modifies:
        - module.weight: Clipped in place to be within the range [min_value, max_value].
        """
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data = torch.clamp(module.weight.data, self.min_value, self.max_value)

    def get_config(self):
        """
        Returns the configuration of the constraint.

        Returns:
        - Dictionary containing min and max values.
        """
        return {'min_value': self.min_value, 'max_value': self.max_value}


class UnitNormConstraint:
    def __call__(self, weights):
        # Normalize weights to unit norm
        return weights / torch.norm(weights)


class StartEndOutputNode(nn.Module):
    def __init__(self, in_features, out_features, initializer=nn.init.xavier_uniform_,max_sequence_length=576):
        super(StartEndOutputNode, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()
        self.max_sequence_length = max_sequence_length
        self.initializer = initializer
        self.unit_norm_constraint = UnitNormConstraint()
        self.constraint = MinMaxValueConstraint(0, self.max_sequence_length)

        # Initialize weights
        self.initializer(self.linear.weight)

    def forward(self, x):
        # Apply the linear layer
        x = self.linear(x)
        # Apply the activation function
        x = self.activation(x)
        # clip value between 0 and max sequences length
        x = self.constraint(x)
        # Apply the unit norm constraint to weights
        self.linear.weight.data = self.unit_norm_constraint(self.linear.weight.data)
        return x

class MutationRateHead(nn.Module):
    def __init__(self, input_dim, min_value=0, max_value=1, initializer=nn.init.xavier_uniform_):
        super(MutationRateHead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Equivalent to Dense(1)
        self.activation = nn.ReLU()  # Equivalent to activation='relu'
        self.constraint = MinMaxValueConstraint(min_value, max_value)
        self.initializer = initializer

        # Apply initializer to the linear weights
        self.initializer(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Apply linear transformation and activation
        x = self.linear(x)
        x = self.activation(x)

        # Apply the constraint to the output values
        x = self.constraint(x)
        return x


class IndelCountHead(nn.Module):
    def __init__(self, input_dim, min_value=0, max_value=50, initializer=nn.init.xavier_uniform_):
        super(IndelCountHead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Fully connected layer equivalent to Dense(1)
        self.activation = nn.ReLU()  # ReLU activation
        self.constraint = MinMaxValueConstraint(min_value, max_value)
        self.initializer = initializer

        # Apply the initializer to the weights
        self.initializer(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Apply the linear transformation and activation
        x = self.linear(x)
        x = self.activation(x)

        # Apply the weight constraint
        x = self.constraint(x)
        return x


class ProductivityHead(nn.Module):
    def __init__(self, input_dim, initializer=nn.init.xavier_uniform_):
        super(ProductivityHead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Fully connected layer equivalent to Dense(1)
        self.activation = nn.Sigmoid()  # Sigmoid activation
        self.initializer = initializer

        # Apply the initializer to the weights
        self.initializer(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Apply the linear transformation and activation
        x = self.linear(x)
        x = self.activation(x)
        return x


class AlleleClassificationHead(nn.Module):
    def __init__(self, input_dim, latent_size_factor, output_dim, activation=nn.SiLU(), initializer=nn.init.xavier_uniform_):
        super(AlleleClassificationHead, self).__init__()
        # First dense layer (mid-layer)
        self.mid_layer = nn.Linear(input_dim, output_dim * latent_size_factor)
        self.activation = activation  # Classification middle layer activation
        # Second dense layer (call head)
        self.call_head = nn.Linear(output_dim * latent_size_factor, output_dim )
        self.sigmoid = nn.Sigmoid()  # Final sigmoid activation
        self.initializer = initializer

        # Apply initializer to weights
        self._initialize_weights()

    def _initialize_weights(self):
        self.initializer(self.mid_layer.weight)
        self.initializer(self.call_head.weight)
        if self.mid_layer.bias is not None:
            nn.init.zeros_(self.mid_layer.bias)
        if self.call_head.bias is not None:
            nn.init.zeros_(self.call_head.bias)

    def forward(self, x):
        x = self.mid_layer(x)
        x = self.activation(x)
        x = self.call_head(x)
        x = self.sigmoid(x)
        return x