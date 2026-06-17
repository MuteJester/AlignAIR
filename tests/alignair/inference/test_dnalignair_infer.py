import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.inference.dnalignair_infer import predict_reads


def test_predict_reads_returns_valid_predictions():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    reads = ["ACGT" * 40, "TTGCAACGTACG" * 8, "ACGTACGTNNACGT" * 6]
    preds = predict_reads(model, rs, reads, batch_size=2)
    assert len(preds) == len(reads)
    vnames, jnames, dnames = (set(rs.gene(g).names) for g in ("V", "J", "D"))
    for p in preds:
        assert p["v_call"] in vnames and p["j_call"] in jnames and p["d_call"] in dnames
        for g in ("v", "d", "j"):
            assert isinstance(p[f"{g}_sequence_start"], int)
            assert 0 <= p[f"{g}_germline_start"] and p[f"{g}_germline_end"] >= 0
