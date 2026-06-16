import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.region_head import REGIONS
from alignair.nn.state_head import STATES


def test_dense_and_scalar_outputs():
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    model = DNAlignAIR(cfg)
    tokens, mask = pad_tokenize(["ACGTACGTACGT", "ACGTAC"])
    out = model.forward_dense(tokens, mask)
    B, L = tokens.shape
    assert out["orientation_logits"].shape == (B, 4)
    assert out["region_logits"].shape == (B, L, len(REGIONS))
    assert out["state_logits"].shape == (B, L, len(STATES))
    for k in ("noise_count", "mutation_rate", "indel_count", "productive"):
        assert out[k].shape == (B, 1)
    # bounded scalars
    assert (out["mutation_rate"] >= 0).all() and (out["mutation_rate"] <= 1).all()
    assert (out["productive"] >= 0).all() and (out["productive"] <= 1).all()
    assert (out["noise_count"] >= 0).all()
    # backbone reps exposed for downstream heads
    assert out["reps"].shape == (B, L, cfg.d_model)


import pytest


def _tiny_refset():
    genairr = pytest.importorskip("GenAIRR")
    import GenAIRR.data as gdata
    from alignair.reference.reference_set import ReferenceSet
    return ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)  # V/J only, smaller


def test_encode_reference_and_match():
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    model = DNAlignAIR(cfg)
    refset = _tiny_refset()
    ref_emb = model.encode_reference(refset)
    assert set(ref_emb) == {"V", "J"}
    nV = len(refset.gene("V").names)
    assert ref_emb["V"]["embeddings"].shape == (nV, cfg.d_model)

    tokens, mask = pad_tokenize(["ACGTACGTACGT", "ACGTAC"])
    out = model(tokens, mask, ref_emb)
    assert out["match"]["V"].shape == (2, nV)
    assert out["match"]["J"].shape == (2, len(refset.gene("J").names))
    # dense outputs still present
    assert out["region_logits"].shape[0] == 2
