import numpy as np
import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.inference.predictor import Predictor


def _model():
    cfg = ModelConfig(max_seq_length=256, v_allele_count=5, j_allele_count=3,
                      d_allele_count=4, has_d_gene=True)
    return SingleChainAlignAIR(cfg), cfg


def test_predict_returns_numpy_dict_with_legacy_keys():
    model, cfg = _model()
    pred = Predictor(model)
    tokens = np.random.randint(0, 6, (6, cfg.max_seq_length))
    out = pred.predict(tokens, batch_size=4)
    for k in ("v_allele", "j_allele", "d_allele", "v_start_logits", "j_end_logits",
              "mutation_rate", "indel_count", "productive"):
        assert k in out
        assert isinstance(out[k], np.ndarray)
    assert out["v_allele"].shape == (6, cfg.v_allele_count)
    assert out["v_start_logits"].shape == (6, cfg.max_seq_length)


def test_predict_batches_match_single_pass():
    model, cfg = _model()
    pred = Predictor(model)
    tokens = np.random.randint(0, 6, (5, cfg.max_seq_length))
    a = pred.predict(tokens, batch_size=2)["v_allele"]
    b = pred.predict(tokens, batch_size=5)["v_allele"]
    # batch-invariance: GPU float reductions differ from CPU at ~1e-6, so use a
    # realistic tolerance (this checks padding/masking correctness, not bit-exactness).
    assert np.allclose(a, b, atol=1e-4)
