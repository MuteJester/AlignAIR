import torch
from alignair.nn.heads.segmentation import (
    SegmentationHead, AlleleClassificationHead, MutationRateHead,
    IndelCountHead, ProductivityHead, ChainTypeHead,
)


def test_segmentation_head_shape():
    head = SegmentationHead(in_features=64, max_seq_length=16)
    feats = torch.randn(2, 64)
    logits = head(feats)
    assert logits.shape == (2, 16)


def test_allele_head_probabilities():
    head = AlleleClassificationHead(in_features=64, latent_dim=20, num_alleles=5,
                                    mid_activation="swish")
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 5)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid


def test_mutation_rate_head_range():
    head = MutationRateHead(in_features=64, max_seq_length=16)
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 1)
    assert (out >= 0).all()


def test_indel_head_shape():
    head = IndelCountHead(in_features=64, max_seq_length=16)
    assert head(torch.randn(2, 64)).shape == (2, 1)


def test_productivity_head_is_prob():
    head = ProductivityHead(in_features=64)
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_chain_type_head_softmax():
    head = ChainTypeHead(in_features=64, max_seq_length=16, num_types=3)
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 3)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)
