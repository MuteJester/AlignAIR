"""End-to-end predict() contract test (untrained model -> structural validity, no crash)."""
import numpy as np
import pytest
import torch


@pytest.mark.slow
def test_predict_end_to_end_contract():
    import GenAIRR.data as gd
    from alignair.config.alignair_config import AlignAIRConfig
    from alignair.models import AlignAIR
    from alignair.predict import PredictConfig, predict
    from alignair.reference.reference_set import ReferenceSet

    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    mcfg = AlignAIRConfig(max_seq_length=576, has_d=True,
                          v_allele_count=len(ref.gene("V")), j_allele_count=len(ref.gene("J")),
                          d_allele_count=len(ref.gene("D")))
    torch.manual_seed(0)
    model = AlignAIR(mcfg)
    seqs = [ref.gene("V").sequences[0], ref.gene("V").sequences[5]]      # two "reads"

    recs = predict(model, seqs, ref, PredictConfig(max_seq_length=576, has_d=True))
    assert len(recs) == 2
    r = recs[0]
    for g in ("v", "d", "j"):
        assert f"{g}_call" in r and isinstance(r[f"{g}_calls"], list)
        assert isinstance(r[f"{g}_sequence_start"], (int, np.integer))
        assert isinstance(r[f"{g}_cigar"], str)
        if r[f"{g}_call"]:
            assert r[f"{g}_call"] in ref.gene(g.upper()).names          # a real reference allele
        assert r[f"{g}_sequence_end"] >= r[f"{g}_sequence_start"]
