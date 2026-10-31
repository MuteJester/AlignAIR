"""The chip's bank compiler: every allele of the reference as a row of one-hot bases on its gene's frame columns."""
import pytest
import torch

from alignair.reference.reference_set import GeneReference, ReferenceSet
from alignair_lab.architectures.chip.bank import BankCompiler

pytestmark = pytest.mark.component


def _reference():
    v = GeneReference(["V*01", "V*02"], ["ACGT", "ACGA"], {"V*01": 0, "V*02": 1},
                      gapped={"V*01": "AC..GT", "V*02": "AC..GA"})
    d = GeneReference(["D*01", "D*02"], ["GGAT", "CCN"], {"D*01": 0, "D*02": 1})
    j = GeneReference(["J*01"], ["TTGG"], {"J*01": 0}, anchors={"J*01": 1})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def _base(table, row, column, width):
    """The row's base at a column ('A', 'T', 'G', 'C') or None, read back from the one-hot table."""
    slots = table[row].reshape(width, 4)[column]
    return "ATGC"[int(slots.argmax())] if slots.sum() else None


def test_rows_are_one_hot_at_their_frame_columns():
    bank = BankCompiler(_reference())
    assert bank.width("v") == 384 and bank.rows("v") == 2
    table, cover = bank.table_v, bank.cover_v
    assert table.shape == (2, 384 * 4) and cover.shape == (2, 384)
    assert [_base(table, 0, c, 384) for c in range(7)] == ["A", "C", None, None, "G", "T", None]
    assert [_base(table, 1, c, 384) for c in range(6)] == ["A", "C", None, None, "G", "A"]
    assert cover[0, :7].tolist() == [1, 1, 0, 0, 1, 1, 0] and cover[0].sum() == 4
    assert table.sum(-1).tolist() == cover.sum(-1).tolist()


def test_d_has_reverse_complement_rows():
    bank = BankCompiler(_reference())
    assert bank.rows("d") == 4
    assert bank.row_allele_d.tolist() == [0, 1, 0, 1] and bank.row_inverted_d.tolist() == [False, False, True, True]
    assert [_base(bank.table_d, 2, c, 48) for c in range(4)] == ["A", "T", "C", "C"]     # reverse complement of GGAT
    assert [_base(bank.table_d, 1, c, 48) for c in range(4)] == ["C", "C", None, None]  # N is no base; D*02 is short


def test_row_mask_maps_alleles_to_both_d_rows():
    bank = BankCompiler(_reference())
    allele_mask = torch.tensor([[True, False], [False, True]])
    assert bank.row_mask("d", allele_mask).tolist() == [[True, False, True, False], [False, True, False, True]]
    assert bank.row_mask("v", allele_mask).tolist() == [[True, False], [False, True]]


def test_no_parameters_and_no_saved_state():
    bank = BankCompiler(_reference())
    assert list(bank.parameters()) == [] and bank.state_dict() == {}


def test_v_and_j_rows_also_hold_their_ungapped_germline_and_anchor_base():
    bank = BankCompiler(_reference())
    assert bank.sequence_v.tolist() == [[1, 4, 3, 2], [1, 4, 3, 1]]     # ACGT, ACGA as tokens (A 1, T 2, G 3, C 4)
    assert bank.sequence_j.tolist() == [[2, 2, 3, 3]]                    # TTGG
    assert bank.anchor_base_v.tolist() == [-1, -1]                       # no annotated anchor
    assert bank.anchor_base_j.tolist() == [1]
