import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.inference.xattn_infer import predict_reads_xattn


def test_predict_reads_xattn_emits_valid_airr_records():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = XAttnAligner(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64))
    reads = ["ACGTACGT" * 30, "TTGCAACGTACG" * 20, "ACGTACGTNNACGT" * 14]
    preds = predict_reads_xattn(model, rs, reads, batch_size=2)
    assert len(preds) == len(reads)
    vn, dn, jn = (set(rs.gene(g).names) for g in ("V", "D", "J"))
    for p in preds:
        assert p["v_call"] in vn and p["d_call"] in dn and p["j_call"] in jn
        assert p["v_calls"] and p["v_call"] in p["v_call_set"]
        for g in ("v", "d", "j"):
            assert isinstance(p[f"{g}_sequence_start"], int)
            assert p[f"{g}_germline_end"] >= 0
        assert "sequence" in p and isinstance(p.get("productive", False), bool)


def test_rescore_query_coords_are_non_negative_spans():
    # the rewrite must derive query coords from region_logits (decode_boundaries), not the untrained
    # boundary head, so sequence_end >= sequence_start (no negative spans) for any gene reported.
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    torch.manual_seed(0)
    model = XAttnAligner(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64))
    reads = ["ACGTACGT" * 30, "TTGCAACGTACG" * 20]
    preds = predict_reads_xattn(model, rs, reads, batch_size=2, rescore=True)
    for p in preds:
        for g in ("v", "d", "j"):
            assert p[f"{g}_sequence_end"] >= p[f"{g}_sequence_start"] >= 0
