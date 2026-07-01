import random, pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.airristotle.tokenizer import AIRRTokenizer
from alignair.airristotle.prompt import build_example
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import build_experiment


def _clean_record():
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, dict(
        mutation_rate=0.0, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
        indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0)))
    return list(exp.stream_records(n=1, seed=1))[0]


def test_copy_target_for_v_start_points_at_true_read_position():
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    rec = _clean_record()
    ex = build_example(rec, rs, tok, n_distractors=4, rng=random.Random(0))
    ids = ex.input_ids
    read_tok = tok.id(tok.READ)
    read_block_start = ids.index(read_tok) + 1
    true_vs = int(rec["v_sequence_start"])
    target_prompt_pos = read_block_start + true_vs
    steps = [t for t in range(len(ids)) if ex.loss_mask[t]]
    assert any(ex.is_copy[t] and ex.copy_target[t] == target_prompt_pos for t in steps)


def test_copy_target_for_v_call_points_at_true_allele_block():
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    rec = _clean_record()
    ex = build_example(rec, rs, tok, n_distractors=4, rng=random.Random(0))
    v_marker = tok.id(tok.V)
    copy_calls = [ex.copy_target[t] for t in range(len(ex.input_ids))
                  if ex.loss_mask[t] and ex.is_copy[t] and ex.input_ids[ex.copy_target[t]] == v_marker]
    assert copy_calls


def test_prompt_len_and_masks_consistent():
    tok = AIRRTokenizer(); rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ex = build_example(_clean_record(), rs, tok, n_distractors=4, rng=random.Random(0))
    n = len(ex.input_ids)
    assert len(ex.gen_target) == len(ex.copy_target) == len(ex.is_copy) == len(ex.loss_mask) == n
    assert all(ex.copy_target[t] < ex.prompt_len for t in range(n) if ex.loss_mask[t] and ex.is_copy[t])
    assert sum(ex.loss_mask) > 0
