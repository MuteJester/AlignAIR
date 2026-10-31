"""Anchor targets: with ``AlignAIRConfig.anchor_heads`` on, the nucleotide adapter gives each read its V anchor (the
first base of the V Cys codon) and J anchor (the first base of the J Trp/Phe codon) as long read positions, or the null
slot L (the model's ``max_seq_length``) when the anchor is not observed.

Each anchor is found on its own. The allele's annotated germline anchor (the reference's ``anchors``, the annotation
the AIRR assembly reads) is mapped through the segment's alignment (its CIGAR, germline and read starts), so an indel
before the anchor moves it with the read. The anchor is observed when its segment is in the read, its defining base is
an aligned (not deleted) base, and its codon lies wholly inside the read the model sees (a crop can cut it). The
defining base is the first base of the V Cys codon and the last base of the J Trp/Phe codon, as the assay's truth
defines the junction. A visible anchor is observed even when the other is not. They are forward-frame positions, like
the boundaries. With the flag off (conv-v1), a batch is as before."""
import itertools

import pytest
import torch

from alignair.core import AlignAIRConfig
from alignair.data import NucleotideBatchAdapter
from alignair.neural.heads.orientation import NUM_ORIENTATIONS, apply_orientation
from alignair.reference.reference_set import GeneReference, ReferenceSet

# a 30-nt read: V [2, 12) with the Cys codon at 9, N1 [12, 14), D [14, 17), N2 [17, 19), J [19, 28) with the Trp
# codon at 21, post [28, 30)
SEQUENCE = "GA" + "CAGGTGCTGT" + "TT" + "GGT" + "AC" + "GCTGGGCAG" + "CC"
SPANS = {"v": (2, 12), "d": (14, 17), "j": (19, 28)}
ANCHORS = {"V": 7, "J": 2}            # germline positions: read 9 = V germline 7, read 21 = J germline 2
NULL = 64


def _segment(gene, start, end, germline_start=0, germline_end=None, cigar=None):
    germline_end = germline_start + end - start if germline_end is None else germline_end
    return {f"{gene}_sequence_start": start, f"{gene}_sequence_end": end, f"{gene}_germline_start": germline_start,
            f"{gene}_germline_end": germline_end, f"{gene}_cigar": cigar or f"{end - start}M"}


def _record(**segments):
    """The read with each gene's segment from ``segments`` (None: the gene is not in the read), else its SPANS one.
    GenAIRR's junction fields are None: the anchors never read them."""
    record = {"sequence": SEQUENCE, "sequence_id": "anchors-read", "productive": None, "mutation_rate": 0.,
              "n_indels": 0, "n_quality_errors": 0, "v_call": "V*01", "d_call": "D*01", "j_call": "J*01",
              "junction_start": None, "junction_end": None}
    for gene, span in SPANS.items():
        given = segments.get(gene, _segment(gene, *span))
        record.update(given if given is not None else {f"{gene}_sequence_start": None, f"{gene}_sequence_end": None})
    return record


def _reference(v_germline=None, j_germline=None, anchors=ANCHORS):
    germline = {"V": v_germline or SEQUENCE[2:12], "D": SEQUENCE[14:17], "J": j_germline or SEQUENCE[19:28]}
    return ReferenceSet({gene: GeneReference([f"{gene}*01", f"{gene}*02"], [germline[gene]] * 2,
                                             {f"{gene}*01": 0, f"{gene}*02": 1},
                                             anchors={f"{gene}*01": anchors[gene], f"{gene}*02": anchors[gene]}
                                             if gene in anchors else None)
                         for gene in ("V", "D", "J")}, has_d=True)


def _config(length=NULL, **options):
    return AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, has_d=True, max_seq_length=length,
                          anchor_heads=True, **options)


def _anchors(record, reference=None, config=None):
    batch = NucleotideBatchAdapter(reference or _reference(), config or _config(), augment_orientation=False)([record])
    assert batch.targets["v_anchor"].dtype == torch.long and batch.targets["v_anchor"].shape == (1,)
    return batch.targets["v_anchor"].item(), batch.targets["j_anchor"].item()


@pytest.mark.component
def test_each_anchor_is_its_alleles_anchor_mapped_through_its_alignment():
    """The read's V Cys codon at 9 and J Trp codon at 21, found from the germline anchors (7 and 2), with GenAIRR's
    junction fields None."""
    assert _anchors(_record()) == (9, 21)


@pytest.mark.component
def test_a_v_anchored_read_without_a_j_gives_its_v_anchor_alone():
    """A V-anchored amplicon read, its 3' end (and J) lost: the V anchor is observed, the J anchor null."""
    assert _anchors(_record(j=None)) == (9, NULL)


@pytest.mark.component
@pytest.mark.parametrize("v", [None, _segment("v", 10, 12, germline_start=8)], ids=["no_v", "v_after_the_cys"])
def test_a_j_anchored_read_missing_its_v_anchor_gives_its_j_anchor_alone(v):
    """A J-anchored read whose 5' end loss took V, or the part of V holding the Cys: J observed, V null."""
    assert _anchors(_record(v=v)) == (NULL, 21)


@pytest.mark.component
@pytest.mark.parametrize("boundary_mode,length,window", [("residue", 12, 12), ("residue", 11, 11),
                                                          ("half_open", 13, 12), ("half_open", 12, 11)])
def test_an_anchor_codon_cut_by_the_window_is_not_observed(boundary_mode, length, window):
    """The window keeps the read's first 12 (or 11) bases: the V codon at 9-11 is whole in 12 and cut in 11; the J
    codon at 21 is outside both. The null slot is L, the model's max_seq_length, in both modes."""
    config = _config(length, boundary_mode=boundary_mode)
    assert _anchors(_record(), config=config) == ((9 if window == 12 else length), length)


@pytest.mark.component
@pytest.mark.parametrize("indel", ["deletion", "insertion"])
def test_an_indel_inside_v_before_the_anchor_moves_it_with_the_read(indel):
    """V's alignment has an indel at germline 3, before the Cys: the anchor is where the alignment puts it in the
    read (9), not at the germline offset (germline 8 with a deleted base, 6 with an inserted one)."""
    read_v = SEQUENCE[2:12]
    if indel == "deletion":           # the germline has one more base, which the read lacks
        germline, anchor, cigar = read_v[:3] + "A" + read_v[3:], 8, "3M1D7M"
    else:                             # the read has one base more, which the germline lacks
        germline, anchor, cigar = read_v[:3] + read_v[4:], 6, "3M1I6M"
    reference = _reference(v_germline=germline, anchors={"V": anchor, "J": 2})
    v = _segment("v", 2, 12, germline_end=len(germline), cigar=cigar)
    assert _anchors(_record(v=v), reference) == (9, 21)


@pytest.mark.component
def test_a_codon_a_recombination_trim_cuts_keeps_its_defining_base():
    """A V whose 3' trim cut its Cys codon after the first base (the read's V ends at 10), and a J whose 5' trim cut
    its Trp codon before the last base (the read's J starts at 23): each anchor is observed where the junction
    starts and ends, as GenAIRR and the assay's truth place it."""
    v = _segment("v", 2, 10)                                      # germline 0..8: the Cys's first base, 7, is in V
    j = _segment("j", 23, 28, germline_start=4)                   # germline 4..9: the Trp's last base, 4, is in J
    assert _anchors(_record(v=v, j=j)) == (9, 21)


@pytest.mark.component
@pytest.mark.parametrize("gene", ["v", "j"])
def test_an_anchor_whose_defining_base_is_deleted_is_not_observed(gene):
    """The read lacks the V Cys codon's first base (or the J Trp codon's last base): a deletion in the alignment."""
    if gene == "v":
        germline = SEQUENCE[2:9] + "T" + SEQUENCE[9:12]           # germline 7, the anchor base, is deleted
        reference, segment = _reference(v_germline=germline), _segment("v", 2, 12, germline_end=11, cigar="7M1D3M")
    else:
        germline = SEQUENCE[19:23] + "G" + SEQUENCE[23:28]        # germline 4, the codon's last base, is deleted
        reference, segment = _reference(j_germline=germline), _segment("j", 19, 28, germline_end=10, cigar="4M1D5M")
    expected = (NULL, 21) if gene == "v" else (9, NULL)
    assert _anchors(_record(**{gene: segment}), reference) == expected


@pytest.mark.component
def test_an_unannotated_allele_has_no_anchor():
    assert _anchors(_record(), _reference(anchors={"V": 7})) == (9, NULL)


@pytest.mark.component
@pytest.mark.parametrize("cigar", ["", "8M"], ids=["no_cigar", "cigar_off_its_spans"])
def test_an_anchor_whose_alignment_cannot_be_mapped_is_not_trained(cigar):
    """A V the record places in the read whose alignment cannot be mapped (no CIGAR, or one that disagrees with its
    read and germline spans) is unscorable: its anchor is left out of the loss (``v_anchor_supervise`` False), not
    trained as "not observed". The J anchor, mapped, is supervised and observed."""
    v = _segment("v", 2, 12, cigar=cigar) if cigar else {**_segment("v", 2, 12), "v_cigar": ""}
    batch = NucleotideBatchAdapter(_reference(), _config(), augment_orientation=False)([_record(v=v), _record()])
    assert batch.targets["v_anchor_supervise"].tolist() == [False, True]
    assert batch.targets["j_anchor_supervise"].tolist() == [True, True]
    assert batch.targets["j_anchor"].tolist() == [21, 21]


@pytest.mark.component
def test_anchor_targets_follow_the_forward_frame_in_every_orientation():
    """Every orientation the adapter draws gives the forward read's anchors: the model's teacher-forced correction
    (its orientation applied again) gives back the forward read, as for the boundaries."""
    reference, config = _reference(), _config()
    torch.manual_seed(0)
    batch = NucleotideBatchAdapter(reference, config)([_record()] * 16)
    orientation = batch.inputs["orientation"]
    assert set(orientation.tolist()) == set(range(NUM_ORIENTATIONS))
    tokens = batch.inputs["tokenized_sequence"]
    forward = NucleotideBatchAdapter(reference, config, augment_orientation=False)([_record()])
    corrected = apply_orientation(tokens, tokens != 0, orientation)
    assert (corrected == forward.inputs["tokenized_sequence"]).all()
    assert batch.targets["v_anchor"].tolist() == [9] * 16 and batch.targets["j_anchor"].tolist() == [21] * 16
    assert not (tokens[orientation == 1] == forward.inputs["tokenized_sequence"]).all()


@pytest.mark.component
def test_conv_v1_has_no_anchor_targets():
    """Off, the flag leaves a batch as it was; on, it adds the two anchor targets and their supervision masks only."""
    off = AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, has_d=True, max_seq_length=NULL)
    assert off.architecture == "conv-v1" and not off.anchor_heads
    without = NucleotideBatchAdapter(_reference(), off, augment_orientation=False)([_record()])
    with_anchors = NucleotideBatchAdapter(_reference(), _config(), augment_orientation=False)([_record()])
    assert "v_anchor" not in without.targets
    assert set(with_anchors.targets) - set(without.targets) == {"v_anchor", "j_anchor", "v_anchor_supervise",
                                                                "j_anchor_supervise"}
    assert all(torch.equal(without.targets[key], with_anchors.targets[key]) for key in without.targets)
    assert all(torch.equal(without.inputs[key], with_anchors.inputs[key]) for key in without.inputs)


@pytest.mark.integration
def test_the_anchors_are_genairrs_junction_ends_and_each_visible_anchor_on_every_component():
    """Live HUMAN_IGH_OGRDB records from every component of the default training mix (300 from each first unselected
    stream, 25 from each full-length component's first selected stream), at the full 576-base window:

    - where GenAIRR gives a junction, the anchors are its ``junction_start`` and ``junction_end - 3`` on every record
      but 2, whose anchor's own defining base an indel deleted (a V CIGAR ``246M1D9M``, a J CIGAR ``16M1D22M``):
      there GenAIRR still places a junction and the anchor is null, as the assay's truth has no junction;
    - where GenAIRR gives none, 208 of the 752 reads still show one anchor, which is observed (none shows both)."""
    import GenAIRR.data as gd
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    config = AlignAIRConfig.from_reference(reference, anchor_heads=True)
    records = []
    for component, streams in enumerate(CurriculumStreamPolicy().substreams_for(gd.HUMAN_IGH_OGRDB)):
        for name, parameters in streams:
            if name in ("unselected-0", "selected-0"):
                count = 300 if name.startswith("unselected") else 25
                records += itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, parameters, 7000 + component), count)
    assert len(records) == 7 * 300 + 4 * 25 and max(len(record["sequence"]) for record in records) <= 576
    batch = NucleotideBatchAdapter(reference, config, augment_orientation=False)(records)
    v_anchors, j_anchors = batch.targets["v_anchor"].tolist(), batch.targets["j_anchor"].tolist()
    deleted, lone, junctionless = [], 0, 0
    for record, v, j in zip(records, v_anchors, j_anchors):
        start = record["junction_start"]
        if start is None:
            junctionless += 1
            lone += (v < 576) + (j < 576) == 1
            assert not (v < 576 and j < 576), record["sequence_id"]
        elif (v, j) != (start, record["junction_end"] - 3):
            deleted.append((record["v_cigar"], record["j_cigar"], v == 576, j == 576))
    assert sorted(deleted) == [("246M1D9M", "17M", True, False), ("291M", "16M1D22M", False, True)]
    assert (junctionless, lone) == (752, 208)
