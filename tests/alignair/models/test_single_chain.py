"""Forward-pass tests for the faithful SingleChainAlignAIR model."""
import torch

from alignair.config.alignair_config import AlignAIRConfig
from alignair.models.single_chain import SingleChainAlignAIR

_COMMON = ["v_start", "v_end", "j_start", "j_end", "v_allele", "j_allele",
           "mutation_rate", "indel_count", "productive"]


def test_singlechain_forward_heavy_and_light():
    L = 256                                          # >=128: the cls towers halve 7x
    for has_d in (True, False):
        cfg = AlignAIRConfig(max_seq_length=L, v_allele_count=30, d_allele_count=8,
                             j_allele_count=6, has_d=has_d)
        m = SingleChainAlignAIR(cfg).eval()
        out = m({"tokenized_sequence": torch.randint(0, 6, (3, L))})
        for k in _COMMON:
            assert k in out, k
        assert out["v_allele"].shape == (3, 30)
        assert out["j_allele"].shape == (3, 6)
        # soft-argmax expectations lie within [0, L-1]
        assert (out["v_start"] >= 0).all() and (out["v_start"] <= L - 1).all()
        assert out["v_start"].shape == (3, 1)
        # D heads present iff has_d
        assert ("d_allele" in out) == has_d
        assert ("d_start" in out) == has_d
        if has_d:
            assert out["d_allele"].shape == (3, 8)


def test_singlechain_logits_present_for_loss():
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=10, d_allele_count=4,
                         j_allele_count=4, has_d=True)
    out = SingleChainAlignAIR(cfg).eval()({"tokenized_sequence": torch.randint(0, 6, (2, 256))})
    for b in ("v_start", "v_end", "d_start", "d_end", "j_start", "j_end"):
        assert out[f"{b}_logits"].shape == (2, 256)


def test_clamp_params_bounds_analysis_heads():
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=10, j_allele_count=4, has_d=False)
    m = SingleChainAlignAIR(cfg)
    m.mutation_rate_head.weight.data.fill_(5.0)
    m.indel_count_head.weight.data.fill_(100.0)
    m.clamp_params()
    assert m.mutation_rate_head.weight.max().item() <= 1.0
    assert m.indel_count_head.weight.max().item() <= 50.0
