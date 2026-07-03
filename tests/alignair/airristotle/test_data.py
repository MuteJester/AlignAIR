"""Gym->example logic (build_v2_example) tested against a synthetic reference, no GenAIRR needed."""
import random

from alignair.reference.reference_set import ReferenceSet
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.data import build_v2_example, make_retrievers, collate


def _dna(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _reference():
    v = {f"IGHV{i}*01": _dna(300, i) for i in range(8)}
    d = {f"IGHD{i}*01": _dna(20, 100 + i) for i in range(4)}
    j = {f"IGHJ{i}*01": _dna(50, 200 + i) for i in range(3)}
    return ReferenceSet.from_genotype({"V": v, "D": d, "J": j})


def _rec(rs, vn, dn, jn):
    V, D, J = rs.gene("V"), rs.gene("D"), rs.gene("J")
    query = V.sequences[V.index[vn]] + D.sequences[D.index[dn]] + J.sequences[J.index[jn]]
    return {"sequence": query, "v_call": vn, "d_call": dn, "j_call": jn}


def test_example_targets_are_copyable_and_shortlist_holds_true_v():
    rs = _reference(); tok = AIRRTokenizer(); r = make_retrievers(rs)
    rec = _rec(rs, "IGHV5*01", "IGHD2*01", "IGHJ1*01")
    ex = build_v2_example(rec, rs, tok, r, v_shortlist=4, has_d=True)
    prompt_dna = tok.decode_seq(ex["input_ids"][:ex["prompt_len"]])
    # true alleles must be copyable from the prompt (shortlist force-included the true V)
    assert rs.gene("V").sequences[rs.gene("V").index["IGHV5*01"]] in prompt_dna
    assert rs.gene("D").sequences[rs.gene("D").index["IGHD2*01"]] in prompt_dna
    assert len(ex["input_ids"]) == len(ex["loss_mask"])
    assert sum(ex["loss_mask"][:ex["prompt_len"]]) == 0 and all(ex["loss_mask"][ex["prompt_len"]:])


def test_ambiguous_v_lists_both_in_target():
    rs = _reference(); tok = AIRRTokenizer(); r = make_retrievers(rs)
    rec = _rec(rs, "IGHV3*01", "IGHD0*01", "IGHJ0*01")
    rec["v_call"] = "IGHV3*01,IGHV6*01"                     # ambiguous V
    ex = build_v2_example(rec, rs, tok, r, v_shortlist=4, has_d=True)
    out_dna = tok.decode_seq(ex["input_ids"][ex["prompt_len"]:])
    V = rs.gene("V")
    assert V.sequences[V.index["IGHV3*01"]] in out_dna
    assert V.sequences[V.index["IGHV6*01"]] in out_dna     # both true V present in the target


def test_collate_pads_and_batches():
    rs = _reference(); tok = AIRRTokenizer(); r = make_retrievers(rs)
    exs = [build_v2_example(_rec(rs, f"IGHV{i}*01", "IGHD1*01", "IGHJ2*01"), rs, tok, r, 4, True)
           for i in (1, 2, 3)]
    batch = collate(exs, tok.id(tok.PAD))
    assert batch["input_ids"].shape == batch["loss_mask"].shape
    assert batch["input_ids"].shape[0] == 3
