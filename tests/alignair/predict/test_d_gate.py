"""The D presence head gates the D call of an AIRR row on its argmax class (``core.d_presence``):

- absent: the D is dropped (no call, no D coordinates), and the row is assembled as a read without a D: the bases
  between V and J are one NP region (np1), and the sequence and germline alignments still line up;
- too short to call: the D call is empty (an abstention) and the D's span is kept;
- callable: the row is the ungated one.

The row carries the class (``d_presence``). The stub wraps the test chip, whose own head it overrides."""

import pytest
import torch

from alignair.core.tokens import TOKEN_DICT

from tests.alignair.predict.golden_rows import FIXTURE, golden_model

pytestmark = pytest.mark.integration

D_FIELDS = ("d_sequence_start", "d_sequence_end", "d_germline_start", "d_germline_end", "d_cigar")
V_J_FIELDS = tuple(f"{g}_{k}" for g in "vj" for k in ("call", "sequence_start", "sequence_end", "germline_start",
                                                       "germline_end", "cigar", "sequence_alignment",
                                                       "germline_alignment"))


@pytest.fixture(scope="module")
def golden():
    return golden_model()


class DPresenceStub(torch.nn.Module):
    """The test chip with its D presence logits replaced: peaked at the class ``classes``
    gives each read (keyed by its encoded bases)."""

    def __init__(self, inner, classes: dict):
        super().__init__()
        self.inner, self.classes = inner, classes
        self.cfg = inner.cfg

    def forward(self, batch):
        out = self.inner(batch)
        tokens = batch["tokenized_sequence"]
        reads = [tuple(tokens[row][tokens[row] != 0].tolist()) for row in range(tokens.shape[0])]
        out["d_presence_logits"] = torch.nn.functional.one_hot(
            torch.tensor([self.classes[read] for read in reads]), 3).float() * 10
        return out


def _encoded(read):
    return tuple(TOKEN_DICT.get(base, TOKEN_DICT["N"]) for base in read.upper())


def _reads():
    """The golden reads whose ungated row is assembled completely with a D (full-length reads)."""
    import json
    return json.loads(FIXTURE.read_text())["reads"][:5]


def _alignments_line_up_as_before(row, before):
    """The gate keeps the sequence and germline alignments' lengths, so they line up wherever the ungated row's did
    (the random-init golden model's own J alignments leave some rows unaligned, gated or not)."""
    lengths = (len(row["sequence_alignment"]), len(row["germline_alignment"]))
    assert lengths == (len(before["sequence_alignment"]), len(before["germline_alignment"]))
    return lengths[0] == lengths[1]


def _rows(golden, klass, **options):
    from alignair.api import predict_sequences
    model, reference = golden
    reads = _reads()
    stub = DPresenceStub(model, {_encoded(read): klass for read in reads})
    return (predict_sequences(model, reference, reads, device="cpu", **options),
            predict_sequences(stub, reference, reads, device="cpu", **options))


def test_an_absent_d_row_has_no_d_and_valid_alignments(golden):
    ungated, gated = _rows(golden, 0)
    aligned = 0
    for before, row in zip(ungated, gated):
        assert before["airr_assembly_status"] == "complete" and before["d_call"]
        assert row["d_presence"] == "absent" and row["airr_assembly_status"] == "complete"
        assert (row["d_call"], row["d_calls"], row["d_likelihoods"]) == ("", [], [])
        assert all(row.get(key) is None for key in D_FIELDS)
        assert all(row[key] == before[key] for key in V_J_FIELDS), [k for k in V_J_FIELDS if row[k] != before[k]]
        sequence = row["sequence"]
        assert row["np1"] == sequence[row["v_sequence_end"]:row["j_sequence_start"]] and not row["np2"]
        assert row["np1_length"] == row["j_sequence_start"] - row["v_sequence_end"]
        aligned += _alignments_line_up_as_before(row, before)
        assert (row["junction"], row["productive"]) == (before["junction"], before["productive"])
    assert aligned >= 2


def test_a_short_d_row_abstains_and_keeps_its_span(golden):
    ungated, gated = _rows(golden, 1)
    aligned = 0
    for before, row in zip(ungated, gated):
        assert row["d_presence"] == "short" and row["airr_assembly_status"] == "complete"
        assert (row["d_call"], row["d_calls"], row["d_likelihoods"]) == ("", [], [])
        assert all(row[key] == before[key] for key in D_FIELDS + V_J_FIELDS)
        assert (row["np1"], row["np2"]) == (before["np1"], before["np2"])
        aligned += _alignments_line_up_as_before(row, before)
        assert (row["junction"], row["productive"]) == (before["junction"], before["productive"])
    assert aligned >= 2


def test_a_short_d_row_reports_no_statistic_of_a_d_germline_it_does_not_name(golden):
    """A short D has no call, so its germline alignment is the read's own bases (the Short-D path): its identity and
    germline alignment would describe the read against itself (identity 1.0). The row leaves them empty, and keeps the
    D's read alignment."""
    ungated, gated = _rows(golden, 1)
    for before, row in zip(ungated, gated):
        assert before["d_identity"] is not None and before["d_germline_alignment"]
        assert row["d_identity"] is None
        assert row["d_germline_alignment"] is None and row["d_germline_alignment_aa"] is None
        assert row["d_sequence_alignment"] == before["d_sequence_alignment"]
        # (a row without a J alignment has no j_identity, gated or not)
        assert row.get("v_identity") == before.get("v_identity") and row.get("j_identity") == before.get("j_identity")


def test_a_callable_d_row_is_the_ungated_row(golden):
    ungated, gated = _rows(golden, 2)
    for before, row in zip(ungated, gated):
        assert row.pop("d_presence") == "callable"
        before.pop("d_presence", None)            # the chip's own head classes its rows too
        assert row == before


@pytest.mark.parametrize("klass,name", [(0, "absent"), (1, "short")])
def test_a_light_record_is_gated_too(golden, klass, name):
    """Without the AIRR assembly, a record carries the class and the gated D."""
    ungated, gated = _rows(golden, klass, airr=False)
    for before, row in zip(ungated, gated):
        assert row["d_presence"] == name and row["d_call"] == "" and before["d_call"]
        assert all((row.get(key) is None) == (klass == 0) or row.get(key) == before.get(key) for key in D_FIELDS)
