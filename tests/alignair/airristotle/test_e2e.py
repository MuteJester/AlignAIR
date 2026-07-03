"""End-to-end: build examples -> train a tiny model -> align back through constrained decode and
recover the true allele. Validates the whole v2 pipeline learns and decodes correctly."""
import random

import torch

from alignair.reference.reference_set import ReferenceSet
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss
from alignair.airristotle.data import build_v2_example, make_retrievers, collate
from alignair.airristotle.infer import align, called_names


def _dna(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _reference():
    v = {f"IGHV{i}*01": _dna(120, i) for i in range(6)}
    d = {f"IGHD{i}*01": _dna(15, 100 + i) for i in range(3)}
    j = {f"IGHJ{i}*01": _dna(30, 200 + i) for i in range(2)}
    return ReferenceSet.from_genotype({"V": v, "D": d, "J": j})


def _rec(rs, vn, dn, jn):
    V, D, J = rs.gene("V"), rs.gene("D"), rs.gene("J")
    q = V.sequences[V.index[vn]] + D.sequences[D.index[dn]] + J.sequences[J.index[jn]]
    return {"sequence": q, "v_call": vn, "d_call": dn, "j_call": jn}


def test_train_then_align_recovers_the_true_alleles():
    torch.manual_seed(0)
    rs = _reference(); tok = AIRRTokenizer(); r = make_retrievers(rs)
    recs = [_rec(rs, f"IGHV{i}*01", f"IGHD{i%3}*01", f"IGHJ{i%2}*01") for i in range(3)]
    exs = [build_v2_example(rec, rs, tok, r, v_shortlist=4, has_d=True) for rec in recs]

    cfg = AIRRConfig(vocab_size=tok.vocab_size, d_model=96, n_layers=2, n_heads=4, n_kv_heads=2,
                     d_ff=192, max_seq=1024)
    m = AIRRistotle(cfg)
    batch = collate(exs, tok.id(tok.PAD))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(300):
        opt.zero_grad()
        loss = airristotle_loss(m(batch["input_ids"]), batch)
        loss.backward(); opt.step()

    m.eval()
    for rec in recs:
        called = called_names(align(m, rec["sequence"], rs, tok, r, v_shortlist=4), rs)
        assert rec["v_call"] in called["V"], f"V: got {called['V']} want {rec['v_call']}"
        assert rec["j_call"] in called["J"], f"J: got {called['J']} want {rec['j_call']}"
