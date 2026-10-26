import numpy as np, pytest
from alignair.aa.infer import class_probs
from alignair.aa.strategic import strategic_setup

pytestmark = pytest.mark.component

def _setup():
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model.eval(); return model, cfg, class_maps

def test_shape_and_normalisation():
    model, cfg, cmaps = _setup()
    p = class_probs(model, ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS", "QVQLQESGPGLVKPSETLSLTCTVSGGSIS"],
                    cfg, "cpu", "v_class_logits", batch=8)
    assert p.shape == (2, cmaps["V"].n_classes()) and np.allclose(p.sum(1), 1.0, atol=1e-5)

def test_empty_input_raises():
    model, cfg, _ = _setup()
    with pytest.raises(ValueError):
        class_probs(model, [], cfg, "cpu", "v_class_logits")

def test_oversized_sequence_rejected_not_truncated():
    model, cfg, _ = _setup()
    with pytest.raises(ValueError):
        class_probs(model, ["A" * (cfg.max_seq_length + 1)], cfg, "cpu", "v_class_logits")

def test_invalid_alphabet_rejected():
    model, cfg, _ = _setup()
    with pytest.raises(ValueError):
        class_probs(model, ["EVQLZZZ123"], cfg, "cpu", "v_class_logits")
