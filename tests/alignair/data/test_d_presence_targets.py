"""D presence targets: for a configuration with a D, the nucleotide adapter gives each read its D's class
(``core.d_presence``): absent (0), too short to call (1) or callable (2), by the D's observed germline bases inside the
window the model sees. The observed germline is the D germline from the first to the last aligned base of its
alignment (CIGAR), internal deletions included and edge gaps left out, as the assay's truth measures a segment's
observed length."""
import itertools

import pytest
import torch

from alignair.core import AlignAIRConfig
from alignair.core.d_presence import ABSENT, CALLABLE, D_CALLABLE_MIN, SHORT
from alignair.data import NucleotideBatchAdapter
from alignair.core.orientation import NUM_ORIENTATIONS, apply_orientation
from alignair.reference.reference_set import GeneReference, ReferenceSet

# a 37-nt read: V [2, 12), N1 [12, 14), D [14, 24), N2 [24, 26), J [26, 35)
SEQUENCE = "GA" + "CAGGTGCTGT" + "TT" + "GTATTACTAT" + "AC" + "GCTGGGCAG" + "CC"
SPANS = {"v": (2, 12), "d": (14, 24), "j": (26, 35)}
GERMLINE = {"V": SEQUENCE[2:12], "D": "A" + SEQUENCE[14:24], "J": SEQUENCE[26:35]}   # D germline: 11 bases


def _record(d="default"):
    """The read with its D alignment from ``d``: (read start, read end, germline start, germline end, CIGAR), or
    None for a read without a D; by default the whole D, germline 1..11."""
    record = {"sequence": SEQUENCE, "sequence_id": "d-read", "productive": None, "mutation_rate": 0.,
              "n_indels": 0, "n_quality_errors": 0, "v_call": "V*01", "d_call": "D*01", "j_call": "J*01"}
    for gene, (start, end) in SPANS.items():
        record.update({f"{gene}_sequence_start": start, f"{gene}_sequence_end": end, f"{gene}_germline_start": 0,
                       f"{gene}_germline_end": end - start, f"{gene}_cigar": f"{end - start}M"})
    alignment = (14, 24, 1, 11, "10M") if d == "default" else d
    if alignment is None:
        record.update({"d_sequence_start": None, "d_sequence_end": None})
    else:
        s, e, gs, ge, cigar = alignment
        record.update({"d_sequence_start": s, "d_sequence_end": e, "d_germline_start": gs, "d_germline_end": ge,
                       "d_cigar": cigar})
    return record


def _reference():
    return ReferenceSet({gene: GeneReference([f"{gene}*01", f"{gene}*02"], [germline] * 2,
                                             {f"{gene}*01": 0, f"{gene}*02": 1})
                         for gene, germline in GERMLINE.items()}, has_d=True)


def _config(length=64, **options):
    return AlignAIRConfig(has_d=True, max_seq_length=length,
                          **options)


def _presence(record, config=None):
    batch = NucleotideBatchAdapter(_reference(), config or _config(), augment_orientation=False)([record])
    target = batch.targets["d_presence"]
    assert target.dtype == torch.long and target.shape == (1,)
    return target.item()


@pytest.mark.component
@pytest.mark.parametrize("d,expected", [
    ("default", CALLABLE),                        # 10 observed germline bases
    ((14, 22, 1, 9, "8M"), CALLABLE),             # exactly D_CALLABLE_MIN
    ((14, 21, 1, 8, "7M"), SHORT),                # one fewer
    ((14, 21, 1, 9, "3M1D4M"), CALLABLE),         # an internal deletion is observed germline: 8
    ((14, 22, 1, 8, "3M1I4M"), SHORT),            # an internal insertion is not: 7
    ((14, 21, 1, 9, "1D7M"), SHORT),              # an edge deletion is unobservable: 7
    ((14, 17, 0, 0, "3I"), ABSENT),               # a D span with no aligned base
    (None, ABSENT),                               # no D in the read
], ids=["10", "8", "7", "internal_deletion", "internal_insertion", "edge_deletion", "no_aligned_base", "no_d"])
def test_d_presence_is_the_ds_observed_germline_length(d, expected):
    assert D_CALLABLE_MIN == 8
    assert _presence(_record(d)) == expected


@pytest.mark.component
@pytest.mark.parametrize("length,expected", [(64, CALLABLE), (19, SHORT), (15, ABSENT)])
def test_d_presence_follows_the_window(length, expected):
    """A window of 18 bases (the model keeps its last slot for an end) shows D's first 4 bases: too short; one of 14
    shows none: absent."""
    assert _presence(_record(), _config(length)) == expected


@pytest.mark.component
@pytest.mark.parametrize("d", [(14, 24, 1, 11, ""), (14, 24, 1, 11, "8M")], ids=["no_cigar", "cigar_off_its_spans"])
def test_a_d_whose_alignment_cannot_be_mapped_is_not_trained(d):
    """A D the record places in the read whose alignment cannot be mapped (no CIGAR, or one that disagrees with the
    D's read and germline spans) is unscorable, as the assay's truth calls a segment it cannot observe: its target is
    left out of the loss (``d_presence_supervise`` False), not trained as absent. A mapped D is supervised."""
    batch = NucleotideBatchAdapter(_reference(), _config(), augment_orientation=False)([_record(d), _record()])
    assert batch.targets["d_presence_supervise"].tolist() == [False, True]
    assert batch.targets["d_presence"].tolist()[1] == CALLABLE


@pytest.mark.component
def test_d_presence_in_every_orientation():
    """Every orientation the adapter draws gives the forward read's D presence, as for the boundaries: the model's
    teacher-forced correction gives back the forward read."""
    reference, config = _reference(), _config()
    torch.manual_seed(0)
    records = [_record()] * 8 + [_record((14, 21, 1, 8, "7M"))] * 8
    batch = NucleotideBatchAdapter(reference, config)(records)
    orientation = batch.inputs["orientation"]
    assert set(orientation.tolist()) == set(range(NUM_ORIENTATIONS))
    tokens = batch.inputs["tokenized_sequence"]
    forward = NucleotideBatchAdapter(reference, config, augment_orientation=False)(records)
    assert (apply_orientation(tokens, tokens != 0, orientation) == forward.inputs["tokenized_sequence"]).all()
    assert forward.targets["d_presence"].tolist() == [CALLABLE] * 8 + [SHORT] * 8
    assert torch.equal(batch.targets["d_presence"], forward.targets["d_presence"])
