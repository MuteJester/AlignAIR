"""Anchor targets: with ``AlignAIRConfig.anchor_heads`` on, the nucleotide adapter gives each read its V anchor (the
first base of the V Cys codon, GenAIRR's ``junction_start``) and J anchor (the first base of the J Trp/Phe codon,
``junction_end - 3``) as long read positions, or the null slot L (the model's ``max_seq_length``) when the anchor is
not observed: GenAIRR gives no junction, or the anchor's codon is not wholly inside the read the model sees (a crop
cuts it). They are forward-frame positions, like the boundaries. With the flag off (conv-v1), a batch is as before."""
import itertools

import pytest
import torch

from alignair.core import AlignAIRConfig
from alignair.data import NucleotideBatchAdapter
from alignair.neural.heads.orientation import NUM_ORIENTATIONS, apply_orientation
from alignair.reference.reference_set import GeneReference, ReferenceSet

# a 30-nt read: V [2, 12) with the Cys codon at 9, N1 [12, 14), D [14, 17), N2 [17, 19), J [19, 28) with the Trp
# codon at 21, post [28, 30); the junction is [9, 24)
SEQUENCE = "GA" + "CAGGTGCTGT" + "TT" + "GGT" + "AC" + "GCTGGGCAG" + "CC"
SPANS = {"v": (2, 12), "d": (14, 17), "j": (19, 28)}
JUNCTION = (9, 24)


def _record(junction=JUNCTION):
    record = {"sequence": SEQUENCE, "sequence_id": "anchors-read", "productive": True, "mutation_rate": 0.,
              "n_indels": 0, "n_quality_errors": 0, "v_call": "V*01", "d_call": "D*01", "j_call": "J*01",
              "junction_start": junction[0], "junction_end": junction[1]}
    for gene, (start, end) in SPANS.items():
        record.update({f"{gene}_sequence_start": start, f"{gene}_sequence_end": end, f"{gene}_germline_start": 0,
                       f"{gene}_germline_end": end - start, f"{gene}_cigar": f"{end - start}M"})
    return record


def _reference():
    return ReferenceSet({gene.upper(): GeneReference([f"{gene.upper()}*01", f"{gene.upper()}*02"],
                                                     [SEQUENCE[start:end]] * 2,
                                                     {f"{gene.upper()}*01": 0, f"{gene.upper()}*02": 1})
                         for gene, (start, end) in SPANS.items()}, has_d=True)


def _config(length=64, **options):
    return AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, has_d=True, max_seq_length=length,
                          **options)


def _anchors(batch):
    return batch.targets["v_anchor"].tolist(), batch.targets["j_anchor"].tolist()


@pytest.mark.component
def test_anchor_targets_are_the_junctions_anchors_as_read_positions():
    batch = NucleotideBatchAdapter(_reference(), _config(anchor_heads=True), augment_orientation=False)([_record()])
    assert batch.targets["v_anchor"].dtype == torch.long and batch.targets["v_anchor"].shape == (1,)
    assert _anchors(batch) == ([9], [21])


@pytest.mark.component
@pytest.mark.parametrize("junction", [(None, None), (-3, 24), (9, 31), (27, 33)],
                         ids=["no_junction", "v_before_read", "j_past_read", "both_past"])
def test_an_anchor_not_wholly_in_the_read_is_the_null_slot(junction):
    """GenAIRR's None junction (an anchor end loss or a trim removed), or an anchor codon outside the read, is not
    observed: the null slot, L = 64. The other anchor keeps its position when its own codon is in the read."""
    batch = NucleotideBatchAdapter(_reference(), _config(anchor_heads=True), augment_orientation=False)(
        [_record(junction)])
    expected = {(None, None): ([64], [64]), (-3, 24): ([64], [21]), (9, 31): ([9], [64]), (27, 33): ([27], [64])}
    assert _anchors(batch) == expected[junction]


@pytest.mark.component
@pytest.mark.parametrize("boundary_mode,length", [("residue", 16), ("half_open", 17)])
def test_anchor_targets_are_null_outside_the_window(boundary_mode, length):
    """A read longer than the window (16 positions; a half-open model keeps its last slot for an end) keeps its first
    16 bases: the V anchor (codon 9-11) stays, the J anchor (codon 21-23) is cut away and becomes the null slot, which
    is L, the model's max_seq_length, in both modes."""
    config = _config(length, anchor_heads=True, boundary_mode=boundary_mode)
    batch = NucleotideBatchAdapter(_reference(), config, augment_orientation=False)([_record()])
    assert batch.inputs["tokenized_sequence"][0].ne(0).sum() == 16
    assert _anchors(batch) == ([9], [length])


@pytest.mark.component
@pytest.mark.parametrize("v_anchor,kept", [(13, True), (14, False)])
def test_an_anchor_codon_the_window_cuts_is_not_observed(v_anchor, kept):
    """In a 16-base window, a codon at 13-15 is whole; one at 14-16 is cut."""
    batch = NucleotideBatchAdapter(_reference(), _config(16, anchor_heads=True), augment_orientation=False)(
        [_record((v_anchor, 24))])
    assert batch.targets["v_anchor"].item() == (v_anchor if kept else 16)


@pytest.mark.component
def test_anchor_targets_follow_the_forward_frame_in_every_orientation():
    """Every orientation the adapter draws gives the forward read's anchors: the model's teacher-forced correction
    (its orientation applied again) gives back the forward read, as for the boundaries."""
    reference, config = _reference(), _config(anchor_heads=True)
    torch.manual_seed(0)
    batch = NucleotideBatchAdapter(reference, config)([_record()] * 16)
    orientation = batch.inputs["orientation"]
    assert set(orientation.tolist()) == set(range(NUM_ORIENTATIONS))
    tokens = batch.inputs["tokenized_sequence"]
    forward = NucleotideBatchAdapter(reference, config, augment_orientation=False)([_record()])
    corrected = apply_orientation(tokens, tokens != 0, orientation)
    assert (corrected == forward.inputs["tokenized_sequence"]).all()
    assert _anchors(batch) == ([9] * 16, [21] * 16)
    assert not (tokens[orientation == 1] == forward.inputs["tokenized_sequence"]).all()


@pytest.mark.component
def test_conv_v1_has_no_anchor_targets():
    """Off, the flag leaves a batch as it was; on, it adds the two anchor targets only."""
    off, on = _config(), _config(anchor_heads=True)
    assert off.architecture == "conv-v1" and not off.anchor_heads
    without = NucleotideBatchAdapter(_reference(), off, augment_orientation=False)([_record()])
    with_anchors = NucleotideBatchAdapter(_reference(), on, augment_orientation=False)([_record()])
    assert "v_anchor" not in without.targets
    assert set(with_anchors.targets) - set(without.targets) == {"v_anchor", "j_anchor"}
    assert all(torch.equal(without.targets[key], with_anchors.targets[key]) for key in without.targets)
    assert all(torch.equal(without.inputs[key], with_anchors.inputs[key]) for key in without.inputs)


@pytest.mark.integration
@pytest.mark.parametrize("boundary_mode,length", [("residue", 300), ("half_open", 301)])
def test_anchor_targets_are_genairrs_junction_anchors_on_live_reads(boundary_mode, length):
    """On live GenAIRR IGH reads of a full-length component and the J-anchored amplicon (5' end loss, so many lose
    their V Cys), cropped to a 300-base window: each anchor is GenAIRR's when its codon is whole in the window, else
    the null slot. Some anchors are null because GenAIRR gives no junction, some because the window cuts them."""
    import GenAIRR.data as gd
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    config = AlignAIRConfig.from_reference(reference, max_seq_length=length, anchor_heads=True,
                                           boundary_mode=boundary_mode)
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    records = [record for component in (components[1], components[4])
               for record in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, component, 3), 16)]
    batch = NucleotideBatchAdapter(reference, config, augment_orientation=False)(records)
    reasons = {"no_junction": 0, "cut": 0, "kept": 0}
    for row, record in enumerate(records):
        start = record["junction_start"]
        for key, position in (("v_anchor", start), ("j_anchor", None if start is None else record["junction_end"] - 3)):
            got = batch.targets[key][row].item()
            if position is None:
                reasons["no_junction"] += 1
                assert got == length
            elif position + 3 > 300:
                reasons["cut"] += 1
                assert got == length
            else:
                reasons["kept"] += 1
                assert got == position
    assert all(count >= 2 for count in reasons.values()), reasons
