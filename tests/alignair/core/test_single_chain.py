import torch
from alignair.core.single_chain import SingleChainAlignAIR


def test_single_chain_is_base(tiny_config_d, dummy_tokens):
    model = SingleChainAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    assert out.chain_type is None  # single chain has no chain_type head
    assert out.v_allele.shape[0] == dummy_tokens.shape[0]
