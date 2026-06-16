import pytest
from alignair.config.model_config import ModelConfig


def test_d_gene_config_fields():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200,
                      j_allele_count=7, d_allele_count=34, has_d_gene=True)
    assert cfg.has_d_gene is True
    assert cfg.d_allele_count == 34
    # default latent dim = count * latent_size_factor when latent size is None
    assert cfg.v_latent_dim == 200 * 2
    assert cfg.j_latent_dim == 7 * 2
    assert cfg.d_latent_dim == 34 * 2


def test_no_d_gene_has_no_d_latent():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200,
                      j_allele_count=7, d_allele_count=None, has_d_gene=False)
    assert cfg.has_d_gene is False
    assert cfg.d_latent_dim is None


def test_explicit_latent_size_overrides_factor():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200, j_allele_count=7,
                      d_allele_count=None, has_d_gene=False,
                      v_allele_latent_size=128)
    assert cfg.v_latent_dim == 128


def test_roundtrip_dict():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200, j_allele_count=7,
                      d_allele_count=34, has_d_gene=True)
    cfg2 = ModelConfig.from_dict(cfg.to_dict())
    assert cfg2 == cfg


def test_d_config_consistency_validation():
    # has_d_gene=True but d_allele_count=None must raise
    with pytest.raises(ValueError):
        ModelConfig(max_seq_length=16, v_allele_count=5, j_allele_count=3,
                    d_allele_count=None, has_d_gene=True)
