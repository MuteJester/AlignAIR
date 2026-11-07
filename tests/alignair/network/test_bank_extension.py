"""A batch may carry novel bank rows (``network.bank.extension_inputs``): the chip reads the bound bank's rows plus
the batch's, in the batch bank's order (``BankCompiler.for_batch``), as it reads the bank of the extended reference,
so training on novel alleles needs no rebinding per batch."""
import copy

import GenAIRR.data as gd
import pytest
import torch

from alignair.network.bank import BankCompiler, extension_inputs
from alignair.reference import annotate
from alignair.reference.frames import gene_frames
from alignair.reference.reference_set import ReferenceSet

from .conftest import toy_reference, with_a_novel_v

pytestmark = pytest.mark.component
BUFFERS = ("table", "cover", "sequence", "row_allele", "anchor_base")


def _view(base, extended, gene="v"):
    return BankCompiler(base).for_batch(extension_inputs(base, extended, gene_frames(extended), gene))


def test_a_batchs_extended_bank_is_the_bank_of_the_extended_reference():
    base, extended = toy_reference(), with_a_novel_v()
    view, whole, bound = _view(base, extended), BankCompiler(extended), BankCompiler(base)
    for name in BUFFERS:
        assert torch.equal(getattr(view, f"{name}_v"), getattr(whole, f"{name}_v")), name
    for name in ("table_d", "cover_d", "row_allele_d", "row_inverted_d", "table_j", "anchor_base_j"):
        assert torch.equal(getattr(view, name), getattr(bound, name)), name
    assert view.rows("v") == 3 and view.width("v") == whole.width("v")
    mask = torch.tensor([[True, False, True]])
    assert torch.equal(view.row_mask("v", mask), whole.row_mask("v", mask))


def test_a_novel_v_of_the_first_locus_lands_inside_a_two_locus_bank():
    base = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGK_OGRDB)
    v = base.gene("V")
    sequence = v.sequences[0]
    novel = sequence[:100] + ("A" if sequence[100] != "A" else "C") + sequence[101:]
    extended = base.extend(annotate(ReferenceSet.from_genotype({"V": {"IGHVF1-G1*99": novel}, "J": {}}), known=base))
    inputs = extension_inputs(base, extended, gene_frames(extended), "v")
    order = inputs["bank_order_v"]
    assert torch.equal(order[:198], torch.arange(198)) and int(order[198]) == 366 and int(order[199]) == 198
    assert inputs["bank_extension_anchor_v"].tolist() == [288]
    view = BankCompiler(base).for_batch(inputs)
    assert tuple(view.table_v.shape) == (367, 1536) and torch.equal(view.table_v, BankCompiler(extended).table_v)


def test_a_batch_without_novel_rows_reads_the_bound_bank(chip):
    model, batch, _ = chip
    bank = model.core.bank
    assert bank.for_batch(batch.inputs) is bank


def test_a_d_extension_is_refused():
    with pytest.raises(ValueError):
        extension_inputs(toy_reference(), with_a_novel_v(), gene_frames(with_a_novel_v()), "d")


def test_the_chip_reads_a_batchs_novel_rows_as_it_reads_the_extended_reference(chip):
    model, batch, _ = chip
    extended = with_a_novel_v()
    other = copy.deepcopy(model)
    other.bind_reference(extended)
    inputs = {key: value for key, value in batch.inputs.items() if key != "bank_mask_v"}
    inputs["bank_mask_v"] = torch.ones(2, 3, dtype=torch.bool)
    extra = extension_inputs(toy_reference(), extended, gene_frames(extended), "v")
    for training in (False, True):
        model.train(training), other.train(training)
        torch.manual_seed(0)
        mine = model({**inputs, **extra})
        torch.manual_seed(0)
        theirs = other(inputs)
        assert set(mine) == set(theirs)
        for key in mine:
            assert torch.equal(mine[key], theirs[key]), (training, key)
