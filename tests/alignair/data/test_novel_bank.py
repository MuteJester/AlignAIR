"""A training batch's bank is the reference extended with its reads' novel alleles (``data.novel_bank``): a novel
read's target is its allele's row; a read that cannot tell a novel allele from its base targets both; a novel allele
equal to a known one is that one; the bank mask always holds each novel allele's base; a batch without novel reads is
as before."""
import pytest
import torch

from alignair.core import prepare_model
from alignair.data import NucleotideBatchAdapter
from alignair.errors import TrainingDataError
from alignair.reference.variants import NOVEL_FIELD, NovelAllele, novel_name

from tests.alignair.network.conftest import OPTIONS, SEQUENCE, toy_record, toy_reference

TODAY = {"bank_mask_d", "bank_mask_j", "bank_mask_v", "orientation", "teacher_coord_d", "teacher_coord_j",
         "teacher_coord_v", "teacher_region", "tokenized_sequence"}


def _adapter(reference=None):
    reference = reference or toy_reference()
    setup = prepare_model(reference, **OPTIONS)
    return NucleotideBatchAdapter(reference, setup.config, augment_orientation=False)


def _novel(base_name="V*01", at=2, to="T", sequence=None):
    base = toy_reference().gene("V")
    sequence = sequence or (lambda s: s[:at] + to + s[at + 1:])(base.sequences[base.index[base_name]])
    return NovelAllele(novel_name(base_name, sequence), base_name, sequence)


def _read(allele, *, read_change=True, at=2, to="T"):
    record = toy_record()
    record["v_call"] = allele.name
    if read_change:
        record["sequence"] = SEQUENCE[:at] + to + SEQUENCE[at + 1:]
    record[NOVEL_FIELD] = {"V": allele.to_dict()}
    return record


@pytest.mark.component
def test_a_read_from_a_novel_allele_targets_its_row():
    allele = _novel()
    batch = _adapter()([_read(allele), toy_record()])
    assert batch.targets["v_allele"].tolist() == [[0., 0., 1.], [1., 0., 0.]]
    assert batch.inputs["bank_order_v"].tolist() == [0, 1, 2]
    assert batch.inputs["bank_extension_anchor_v"].tolist() == [4]
    assert "bank_order_j" not in batch.inputs and "bank_order_d" not in batch.inputs


@pytest.mark.component
def test_a_read_that_shows_none_of_its_novel_alleles_snps_also_targets_the_base():
    allele = _novel(at=9, to="A")                      # outside the reads' V germline span [0, 8)
    batch = _adapter()([_read(allele, read_change=False), toy_record()])
    assert batch.targets["v_allele"].tolist() == [[1., 0., 1.], [1., 0., 1.]]


@pytest.mark.component
def test_a_novel_allele_equal_to_a_known_allele_is_that_allele():
    allele = _novel(sequence="TTTTTTTTTT")             # V*02's sequence
    batch = _adapter()([_read(allele, read_change=False)])
    assert batch.targets["v_allele"].tolist() == [[0., 1.]] and "bank_order_v" not in batch.inputs


@pytest.mark.component
def test_the_bank_mask_always_holds_each_novel_alleles_base():
    allele = _novel("V*02", at=8, to="A")
    adapter = _adapter()
    for seed in range(30):
        torch.manual_seed(seed)
        batch = adapter([_read(allele, read_change=False), toy_record()])
        assert bool(batch.inputs["bank_mask_v"][0, 1]), seed


@pytest.mark.component
def test_a_batch_without_novel_reads_has_no_extension():
    batch = _adapter()([toy_record(), toy_record()])
    assert set(batch.inputs) == TODAY and batch.targets["v_allele"].shape[-1] == 2


@pytest.mark.component
def test_one_novel_name_with_two_sequences_in_a_batch_is_refused():
    allele = _novel()
    other = NovelAllele(allele.name, allele.base, allele.sequence[:5] + "G" + allele.sequence[6:])
    with pytest.raises(TrainingDataError) as caught:
        _adapter()([_read(allele), _read(other)])
    assert allele.name in caught.value.context["alleles"]


@pytest.mark.component
def test_a_novel_allele_of_a_base_the_reference_lacks_is_refused():
    allele = _novel()
    lost = NovelAllele(novel_name("V*09", allele.sequence), "V*09", allele.sequence)
    with pytest.raises(TrainingDataError):
        _adapter()([_read(lost)])


@pytest.mark.component
def test_a_novel_d_allele_is_refused():
    record = toy_record()
    record[NOVEL_FIELD] = {"D": NovelAllele("D*01_nv000000000000", "D*01", "GGGCCA", "D").to_dict()}
    with pytest.raises(TrainingDataError):
        _adapter()([record])


def _two_locus_novel_batch():
    """(setup, batch): an IGH + IGK tiny chip's setup and its adapter's batch of 8 IGH reads, the first relabelled as
    read from a novel V allele (a SNP variant of its call)."""
    from itertools import islice

    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    from alignair.train.records.seeds import seed_block
    from alignair.train.records.streams import CurriculumStreamPolicy, stream_records
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGK_OGRDB)
    parameters = dict(CurriculumStreamPolicy().substreams_for(gd.HUMAN_IGH_OGRDB)[1])["unselected-0"]
    records = [dict(r) for r in islice(stream_records(gd.HUMAN_IGH_OGRDB, parameters,
                                                      seed_block(0, "training", 0, 0, 1, "unselected-0")), 8)]
    first = records[0]
    v = reference.gene("V")
    call = first["v_call"].split(",")[0]
    base = v.sequences[v.index[call]]
    p = first["v_germline_start"] + 10
    sequence = base[:p] + ("A" if base[p] != "A" else "C") + base[p + 1:]
    allele = NovelAllele(novel_name(call, sequence), call, sequence)
    first["v_call"], first[NOVEL_FIELD] = allele.name, {"V": allele.to_dict()}
    first["recalled_v_call"] = allele.name            # GenAIRR re-calls a novel read as its novel allele alone
    setup = prepare_model(reference, embed_dim=4, trunk_channels=8)
    return setup, NucleotideBatchAdapter(reference, setup.config, augment_orientation=False)(records)


@pytest.mark.integration
def test_a_two_locus_batch_puts_a_novel_reads_target_in_its_locus():
    _, batch = _two_locus_novel_batch()
    assert batch.targets["v_allele"][0].nonzero().flatten().tolist() == [198]
    assert int(batch.inputs["bank_order_v"][198]) == 366


@pytest.mark.integration
def test_a_two_locus_chip_validates_a_batch_with_novel_rows():
    """Validation runs the chip in eval mode on the adapter's batches, novel rows included: the locus step
    (``refine.locus``) is for prediction batches only, so the batch's extended bank never meets the bound bank's locus
    masks, and the validation metrics read the scans as training does."""
    import torch
    setup, batch = _two_locus_novel_batch()
    torch.manual_seed(0)
    model = setup.build_model().eval()
    with torch.no_grad():
        out = model(batch.inputs)
    assert out["v_allele"].shape[1] == batch.targets["v_allele"].shape[1] and "read_locus" not in out
