import pytest, torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle
from alignair.airristotle.infer import decode_record
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import build_experiment


def _clean_record():
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, dict(
        mutation_rate=0.0, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
        indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0)))
    return list(exp.stream_records(n=1, seed=2))[0]


def test_decode_produces_valid_record_structure():
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    rec = _clean_record()
    cfg = AIRRConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                     d_ff=128, max_seq=8192)
    m = AIRRistotle(cfg).eval()
    out = decode_record(m, tok, rec, rs, n_distractors=4, rng=__import__("random").Random(0), device="cpu")
    # keys present
    for k in ("v_call", "d_call", "j_call", "v_sequence_start", "v_sequence_end",
              "v_germline_start", "v_germline_end", "junction_start", "junction_end",
              "productive", "orientation_id"):
        assert k in out, k
    # calls resolve to real allele names of the right gene
    assert out["v_call"] in rs.gene("V").names
    assert out["j_call"] in rs.gene("J").names
    # read coords are within the read
    L = len(out["sequence"])
    for k in ("v_sequence_start", "v_sequence_end", "junction_start", "junction_end"):
        assert 0 <= out[k] <= L
