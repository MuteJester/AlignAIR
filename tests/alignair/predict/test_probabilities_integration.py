"""Both input alphabets use the same categorical multi-head inference API."""
import numpy as np
import pytest
import torch

from alignair.core import AlignAIR, build_model_config
from alignair.data import SequenceInputAdapter
from alignair.data.tokenizer import AAProteinTokenizer, CenterPaddedTokenizer
from alignair.predict import class_probabilities

pytestmark = pytest.mark.integration


@pytest.mark.parametrize('input_space', ['nucleotide','protein'])
def test_native_model_multi_head_probabilities_preserve_parameters_and_module_modes(input_space):
    cfg = build_model_config(3,2,input_space=input_space,label_space='protein_classes',has_d=False,
                             max_seq_length=192,embed_dim=4,filters=4,block_out=8)
    model = AlignAIR(cfg)
    model.branches['v'].eval()
    modes = [module.training for module in model.modules()]
    before = {key:value.clone() for key,value in model.state_dict().items()}
    sequences = ['ACGT','NNTA','TGCAT'] if input_space == 'nucleotide' else ['ACDEF','GHIK','X*']
    tokenizer = CenterPaddedTokenizer() if input_space == 'nucleotide' else AAProteinTokenizer()
    adapter = SequenceInputAdapter(tokenizer,192,vocab_size=cfg.vocab_size)
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    try:
        output = class_probabilities(model,sequences,adapter,keys=('v_class_logits','j_class_logits'),batch_size=2)
    finally:
        handle.remove()
    assert len(calls) == 2
    assert output['v_class_logits'].shape == (3,3) and output['j_class_logits'].shape == (3,2)
    assert all(np.allclose(value.sum(1),1.) for value in output.values())
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())
