import torch
from alignair.nn.conv import Conv1dBatchNorm, ConvResidualFeatureExtractor


def test_conv_bn_halves_length_and_sets_channels():
    # Input (B, C_in, L); block applies 3 same-conv then maxpool(2).
    block = Conv1dBatchNorm(in_channels=8, filters=16, kernel=3, max_pool=2,
                            activation="leaky_relu")
    x = torch.randn(2, 8, 16)
    out = block(x)
    assert out.shape == (2, 16, 8)  # channels->16, length 16//2=8


def test_conv_bn_uses_eps_and_momentum():
    block = Conv1dBatchNorm(in_channels=4, filters=4, kernel=3, max_pool=2,
                            activation="leaky_relu")
    assert abs(block.batch_norm.eps - 0.8) < 1e-9
    assert abs(block.batch_norm.momentum - 0.9) < 1e-9


def test_residual_extractor_output_shape():
    # embeddings (B, L, E) -> block transposes internally -> (B, out_features)
    fe = ConvResidualFeatureExtractor(
        in_channels=32, filter_size=16,
        kernel_sizes=[3, 3, 3, 2, 5], max_pool_size=2,
        out_features=64, activation="tanh",
    )
    x = torch.randn(2, 64, 32)  # (B, L, E)
    out = fe(x)
    assert out.shape == (2, 64)


def test_residual_extractor_backprop():
    fe = ConvResidualFeatureExtractor(
        in_channels=32, filter_size=16, kernel_sizes=[3, 3, 3, 2, 5],
        max_pool_size=2, out_features=64, activation="tanh",
    )
    x = torch.randn(2, 64, 32, requires_grad=True)
    out = fe(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
