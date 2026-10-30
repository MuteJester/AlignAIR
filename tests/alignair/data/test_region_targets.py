"""Region targets: with ``AlignAIRConfig.region_head`` on, the nucleotide adapter gives each read position its
recombination region (pre, V, N1, D, N2, J, post) as a region-head class, in the forward frame the model's orientation
correction restores, cropped to the model's window as the boundary targets are, the padding value past the read. With
the flag off (conv-v1), a batch is as before: no region target, no ``regions`` weight."""
import itertools

import pytest
import torch

from alignair.core import AlignAIRConfig
from alignair.data import NucleotideBatchAdapter
from alignair.neural.heads.orientation import NUM_ORIENTATIONS, apply_orientation
from alignair.neural.heads.region import READ_REGIONS
from alignair.reference.reference_set import GeneReference, ReferenceSet
from alignair.tasks.losses import loss_weight_names

CLASS = {name: k for k, name in enumerate(READ_REGIONS)}
# a 30-nt read: pre [0, 2), V [2, 12), N1 [12, 14), D [14, 17), N2 [17, 19), J [19, 28), post [28, 30)
SEQUENCE = "GA" + "CAGGTGCAGC" + "TT" + "GGT" + "AC" + "TGGGGCCAG" + "CC"
SPANS = {"v": (2, 12), "d": (14, 17), "j": (19, 28)}
REGIONS = (["pre"] * 2 + ["V"] * 10 + ["N1"] * 2 + ["D"] * 3 + ["N2"] * 2 + ["J"] * 9 + ["post"] * 2)
EXPECTED = [CLASS[name] for name in REGIONS]


def _record():
    record = {"sequence": SEQUENCE, "sequence_id": "regions-read", "productive": True, "mutation_rate": 0.,
              "n_indels": 0, "n_quality_errors": 0, "v_call": "V*01", "d_call": "D*01", "j_call": "J*01"}
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


@pytest.mark.component
def test_region_targets_are_each_read_positions_region_as_a_head_class():
    assert READ_REGIONS == ("pre", "V", "N1", "D", "N2", "J", "post")
    batch = NucleotideBatchAdapter(_reference(), _config(region_head=True), augment_orientation=False)([_record()])
    labels = batch.targets["region_labels"]
    assert labels.shape == (1, 64) and labels.dtype == torch.long
    assert labels[0, :30].tolist() == EXPECTED and (labels[0, 30:] == -100).all()


@pytest.mark.component
def test_region_targets_follow_the_forward_frame_in_every_orientation():
    """Every orientation the adapter draws gives the same region targets, those of the forward read: the model's
    teacher-forced correction (its orientation applied again) gives back the forward read, as for the boundaries."""
    reference, config = _reference(), _config(region_head=True)
    forward = NucleotideBatchAdapter(reference, config, augment_orientation=False)([_record()])
    torch.manual_seed(0)
    batch = NucleotideBatchAdapter(reference, config)([_record()] * 16)
    orientation = batch.inputs["orientation"]
    assert set(orientation.tolist()) == set(range(NUM_ORIENTATIONS))
    tokens = batch.inputs["tokenized_sequence"]
    corrected = apply_orientation(tokens, tokens != 0, orientation)
    for k in range(NUM_ORIENTATIONS):
        rows = orientation == k
        assert (batch.targets["region_labels"][rows] == forward.targets["region_labels"]).all(), k
        assert (corrected[rows] == forward.inputs["tokenized_sequence"]).all(), k
        for gene in SPANS:
            assert (batch.targets[f"{gene}_start"][rows] == forward.targets[f"{gene}_start"]).all(), (k, gene)
    assert not (tokens[orientation == 1] == forward.inputs["tokenized_sequence"]).all()


@pytest.mark.component
@pytest.mark.parametrize("boundary_mode,length", [("residue", 16), ("half_open", 17)])
def test_region_targets_are_windowed_like_the_boundary_targets(boundary_mode, length):
    """A read longer than the window (16 positions; a half-open model keeps its last slot for an end) keeps its first
    16 regions, cut where its boundary targets are cut: D ends at 16, J is outside it."""
    config = _config(length, region_head=True, boundary_mode=boundary_mode)
    batch = NucleotideBatchAdapter(_reference(), config, augment_orientation=False)([_record()])
    labels, targets = batch.targets["region_labels"][0], batch.targets
    assert labels.tolist() == EXPECTED[:16] + [-100] * (length - 16)
    assert (targets["d_start"].item(), targets["d_end"].item()) == (14, 16) and not targets["j_supervise"].item()
    assert (labels[14:16] == CLASS["D"]).all()


@pytest.mark.component
def test_conv_v1_has_no_regions_weight_or_target():
    """Off, the flag leaves a batch as it was: the same targets, none for regions, and no ``regions`` weight; on, it
    adds the region target and the weight only."""
    off, on = _config(), _config(region_head=True)
    assert off.architecture == "conv-v1" and not off.region_head
    assert "regions" not in loss_weight_names(off)
    assert loss_weight_names(on) == (*loss_weight_names(off), "regions")
    without = NucleotideBatchAdapter(_reference(), off, augment_orientation=False)([_record()])
    with_regions = NucleotideBatchAdapter(_reference(), on, augment_orientation=False)([_record()])
    assert "region_labels" not in without.targets
    assert set(with_regions.targets) - set(without.targets) == {"region_labels"}
    assert all(torch.equal(without.targets[key], with_regions.targets[key]) for key in without.targets)
    assert all(torch.equal(without.inputs[key], with_regions.inputs[key]) for key in without.inputs)


@pytest.mark.integration
@pytest.mark.parametrize("boundary_mode,length", [("residue", 300), ("half_open", 301)])
def test_region_targets_agree_with_the_boundary_targets_on_genairr_reads(boundary_mode, length):
    """On live GenAIRR IGH reads (most longer than the 300-position window, so prefix-cropped), the region targets
    cover the encoded read exactly, and each supervised gene's boundary span holds that gene's class only, including
    a span the window cuts short."""
    import GenAIRR.data as gd
    from alignair.train.gym import Curriculum
    from alignair.train.gym.streams import stream_records
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    config = AlignAIRConfig.from_reference(reference, max_seq_length=length, region_head=True,
                                           boundary_mode=boundary_mode)
    records = list(itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, dict(Curriculum().params(0.6)), 0), 12))
    batch = NucleotideBatchAdapter(reference, config, augment_orientation=False)(records)
    tokens, labels = batch.inputs["tokenized_sequence"], batch.targets["region_labels"]
    assert ((labels != -100) == (tokens != 0)).all()
    assert sum(len(record["sequence"]) > 300 for record in records) >= 6
    assert batch.targets["v_supervise"].all()
    cut = 0
    for row, record in enumerate(records):
        for gene in ("v", "d", "j"):
            if batch.targets[f"{gene}_supervise"][row]:
                start, end = int(batch.targets[f"{gene}_start"][row]), int(batch.targets[f"{gene}_end"][row])
                assert (labels[row, start:end] == CLASS[gene.upper()]).all(), (row, gene)
                cut += end == 300 < record[f"{gene}_sequence_end"]
    assert cut >= 1
