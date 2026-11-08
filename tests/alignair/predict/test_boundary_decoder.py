"""The product reports each boundary as the argmax of the masked position logits, not their soft expectation."""
import numpy as np
import pytest
import torch

from alignair.predict.clean import clean
from alignair.predict.outputs import ModelOutputs
from alignair.predict.decode import decode_boundaries, decoded_boundaries
from alignair.predict.segment import correct_segments

pytestmark = pytest.mark.component

NEG = -1e4


def _row(length=12, **peaks):
    row = torch.full((1, length), NEG)
    for position, value in peaks.items():
        row[0, int(position[1:])] = value
    return row


def test_boundaries_are_the_argmax_of_the_masked_logits():
    # two modes, at 2 and 10: the soft expectation lands between them, the argmax on the taller one
    logits = _row(p2=4.0, p10=5.0)
    mask = torch.ones(1, 12, dtype=torch.bool)
    expectation = (torch.softmax(logits, -1) * torch.arange(12)).sum().item()
    assert 3 < expectation < 9
    start, end = decode_boundaries(logits, logits.flip(-1), mask, mask)
    assert start.tolist() == [10] and end.tolist() == [1] and start.dtype == torch.long


def test_padding_never_wins_the_argmax():
    logits = _row(p3=1.0, p9=50.0)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, :6] = True
    start, end = decode_boundaries(logits, logits, mask, mask)
    assert start.tolist() == [3] and end.tolist() == [3]


def test_ties_take_the_first_position():
    logits = torch.zeros(2, 8)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, :2] = False
    start, end = decode_boundaries(logits, logits, mask, mask)
    assert start.tolist() == [0, 2] and end.tolist() == [0, 2]


def test_a_one_hot_row_decodes_to_the_coordinate_the_floored_expectation_gave():
    length, seq_len = 20, 11
    lens = np.array([seq_len])
    for k in (0, 1, 7, seq_len - 1):
        logits = _row(length, **{f"p{k}": 60.0})
        expectation = (torch.softmax(logits, -1) * torch.arange(length)).sum(-1, keepdim=True)
        outputs = {"v_start_logits": logits, "v_end_logits": logits, "v_start": expectation, "v_end": expectation,
                   "start_mask": torch.arange(length)[None] < seq_len,
                   "end_mask": torch.arange(length)[None] < seq_len + 1}
        start, end = decoded_boundaries(outputs, "v")
        soft, hard = expectation.numpy().reshape(-1), start.numpy()
        old = correct_segments({"v": soft, "j": soft}, {"v": soft, "j": soft}, lens, length)
        new = correct_segments({"v": hard, "j": hard}, {"v": end.numpy(), "j": end.numpy()}, lens, length)
        assert new.start["v"].tolist() == old.start["v"].tolist() == [k]
        assert new.end["v"].tolist() == old.end["v"].tolist()
        assert new.start["j"].tolist() == old.start["j"].tolist()


def test_half_open_masks_are_used_as_given():
    logits = _row(p3=2.0, p6=9.0)
    outputs = {"j_start_logits": logits, "j_end_logits": logits, "start_mask": torch.arange(12)[None] < 5,
               "end_mask": torch.arange(12)[None] < 7}
    start, end = decoded_boundaries(outputs, "j")
    assert start.tolist() == [3] and end.tolist() == [6]


def test_clean_reports_the_argmax_and_keeps_the_expectation_without_logits():
    logits = _row(p2=4.0, p10=5.0)
    batch = {"v_allele": np.array([[0.1, 0.9]]), "v_start": np.array([[6.4]]), "v_end": np.array([[7.2]]),
             "v_start_logits": logits.numpy(), "v_end_logits": logits.numpy(), "start_mask": np.ones((1, 12), bool),
             "end_mask": np.ones((1, 12), bool),
             "j_allele": np.array([[0.5, 0.5]]), "j_start": np.array([[8.0]]), "j_end": np.array([[9.0]]),
             "mutation_rate": np.array([[0.1]]), "indel_count": np.array([[0.0]])}
    preds = clean([ModelOutputs.from_network(batch, ("v", "j"))], genes=("v", "j"))
    assert preds.start["v"].tolist() == [10] and preds.end["v"].tolist() == [10]
    assert preds.start["j"].tolist() == [8.0]


def test_a_non_finite_logit_on_an_allowed_slot_is_refused():
    from alignair.errors import NumericalError
    mask = torch.ones(1, 6, dtype=torch.bool)
    for bad in (float("nan"), float("inf")):
        logits = torch.zeros(1, 6)
        logits[0, 2] = bad
        with pytest.raises(NumericalError, match="non-finite boundary logit"):
            decode_boundaries(logits, torch.zeros(1, 6), mask, mask)
    # a non-finite logit on a pad slot is never decoded
    logits = torch.zeros(1, 6)
    logits[0, 5] = float("nan")
    short = mask.clone()
    short[0, 5] = False
    assert decode_boundaries(logits, logits, short, short)[0].tolist() == [0]
