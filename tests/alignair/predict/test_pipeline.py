"""End-to-end predict() contract test (untrained model -> structural validity, no crash)."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.component


@pytest.mark.slow
def test_predict_end_to_end_contract():
    from alignair.model_file import load_model
    from alignair.predict import PredictConfig, predict
    from tests.alignair.fixtures.build_test_chip import TEST_CHIP

    loaded = load_model(str(TEST_CHIP))
    model, ref = loaded.model.eval(), loaded.reference
    seqs = [ref.gene("V").sequences[0], ref.gene("V").sequences[5]]      # two "reads"

    recs = predict(model, seqs, ref, PredictConfig(max_seq_length=model.cfg.max_seq_length, has_d=True))
    assert len(recs) == 2
    r = recs[0]
    for g in ("v", "d", "j"):
        assert f"{g}_call" in r and isinstance(r[f"{g}_calls"], list)
        if r[f"{g}_call"]:
            assert set(r[f"{g}_call"].split(",")) <= set(ref.gene(g.upper()).names)   # real reference alleles
        if f"{g}_sequence_start" in r:
            assert isinstance(r[f"{g}_sequence_start"], (int, np.integer)) and isinstance(r[f"{g}_cigar"], str)
            assert r[f"{g}_sequence_end"] >= r[f"{g}_sequence_start"]
    assert r["v_call"] and r["v_sequence_start"] == 0          # a germline V read is called from its first base
