"""The AIRR rows of a model with anchor heads take their junction, frame, stop codon and productivity from the decoded
anchors (``predict.junction.derive_junction``, over the row's own V start and J end); the test chip's rows are the
rows recorded for it.

A decoded anchor is the argmax of its logits over the read's positions and the null slot L; the null slot means the
anchor is not observed (None)."""
import itertools
import json

import pytest
import torch

from alignair.data.projection import translate_nt
from alignair.core.tokens import TOKEN_DICT
from alignair.predict.decode import decode_anchors
from alignair.predict.junction import derive_junction

from tests.alignair.predict.golden_rows import FIXTURE, golden_model, predict_rows

JUNCTION_KEYS = ("junction", "junction_aa", "junction_length", "junction_aa_length", "junction_start", "junction_end",
                 "cdr3", "cdr3_aa", "cdr3_start", "cdr3_end")


@pytest.fixture(scope="module")
def golden():
    return golden_model()


@pytest.mark.component
def test_anchors_decode_by_argmax_over_the_codon_starts_and_the_null_slot():
    """A 6-base read in a window of 8: an anchor is a codon start, 0..3. The last two read positions (4, 5), where no
    whole codon fits, and the pads never win; the null slot (L) can; a tie takes the first slot; the result is a long
    position, L for the null slot."""
    read = torch.zeros(5, 8, dtype=torch.bool)
    read[:, :6] = True
    logits = torch.full((5, 9), -5.0)
    logits[0, 2] = 3.0                       # a codon start
    logits[1, 7] = 9.0                       # a pad: never decoded
    logits[1, 1] = 1.0
    logits[2, 8] = 4.0                       # the null slot
    logits[3, [1, 3]] = 2.0                  # a tie
    logits[4, [4, 5]] = 9.0                  # read positions too near the end for a codon: never decoded
    logits[4, 0] = 1.0
    assert decode_anchors(logits, read).tolist() == [2, 1, 8, 1, 0]


@pytest.mark.component
def test_anchors_refuse_a_non_finite_logit_on_an_allowed_slot():
    """A NaN on a codon start is refused; one where no codon fits (the read's last two positions) is never read."""
    from alignair.errors import NumericalError
    read = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(NumericalError, match="non-finite anchor logit"):
        decode_anchors(torch.tensor([[0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]]), read)
    assert decode_anchors(torch.tensor([[0.0, 1.0, 0.0, float("nan"), float("nan"), 0.0]]), read).tolist() == [1]


@pytest.mark.integration
def test_the_test_chips_rows_are_unchanged(golden):
    """The test chip's rows on the golden reads (full length, with and without a V indel, J-anchored amplicons, a
    read longer than the window, a reverse-complemented read) are the rows recorded for it, each with its anchors."""
    fixture = json.loads(FIXTURE.read_text())
    rows = predict_rows(*golden, fixture["reads"])
    assert rows == fixture["rows"]
    assert all("v_anchor" in row and "j_anchor" in row for row in rows)


class AnchorStub(torch.nn.Module):
    """The test chip with its anchors given: its outputs (less a learned ``productive``, should it have one),
    the orientation ``orientations`` gives each read (the forward one, 0, by default), and V and J anchor logits peaked
    at the positions ``anchors`` gives each read (None: the null slot). Both are keyed by the read's encoded bases as
    presented; the anchors are positions in the forward frame the orientation corrects the read to, as a model's
    heads read the corrected read."""

    def __init__(self, inner, anchors: dict, orientations: dict | None = None):
        super().__init__()
        self.inner, self.anchors, self.orientations = inner, anchors, orientations or {}
        self.cfg = inner.cfg

    def forward(self, batch):
        out = self.inner(batch)
        out.pop("productive", None)
        tokens = batch["tokenized_sequence"]
        rows, length = tokens.shape
        reads = [tuple(tokens[row][tokens[row] != 0].tolist()) for row in range(rows)]
        out["orientation"] = torch.tensor([self.orientations.get(read, 0) for read in reads], dtype=torch.long)
        out["orientation_logits"] = torch.nn.functional.one_hot(out["orientation"], 4).float() * 10
        for k, key in enumerate(("v_anchor_logits", "j_anchor_logits")):
            logits = torch.zeros(rows, length + 1)
            for row, read in enumerate(reads):
                position = self.anchors[read][k]
                logits[row, length if position is None else position] = 20.0
            out[key] = logits
        return out


def _encoded(read):
    return tuple(TOKEN_DICT.get(base, TOKEN_DICT["N"]) for base in read.upper())


@pytest.mark.integration
def test_an_anchor_model_row_takes_its_junction_from_the_anchors(golden):
    """Given a GenAIRR read's true anchors, the row's junction is GenAIRR's, its translation the junction's
    (GenAIRR's convention), its CDR3 the junction without the anchor codons, and its frame, stop codon and
    productivity the rule's over the row's own V start and J end; ``productive_prediction`` is the same derived
    call, since no head learned it. An anchor the stub calls not observed, or reversed anchors, give a row whose
    junction and flags are unknown, assembled without error."""
    import GenAIRR.data as gd
    from alignair.api import predict_sequences
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    model, reference = golden
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    records = [r for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[1], 31), 12)
               if r["junction_start"] is not None][:6]
    reads = [r["sequence"].upper() for r in records]
    anchors = {_encoded(read): (r["junction_start"], r["junction_end"] - 3) for read, r in zip(reads, records)}
    anchors[_encoded(reads[4])] = (None, anchors[_encoded(reads[4])][1])                 # V not observed
    anchors[_encoded(reads[5])] = (anchors[_encoded(reads[5])][1] + 3, anchors[_encoded(reads[5])][0])   # reversed
    rows = predict_sequences(AnchorStub(model, anchors), reference, reads, device="cpu", batch_size=4)
    for k, (row, record) in enumerate(zip(rows, records)):
        assert row["airr_assembly_status"] != "failed", row.get("airr_assembly_error")
        v_anchor, j_anchor = anchors[_encoded(reads[k])]
        assert (row["v_anchor"], row["j_anchor"]) == (v_anchor, j_anchor)
        call = derive_junction(row["sequence"], v_anchor, j_anchor,
                               coding_span=(row.get("v_sequence_start"), row.get("j_sequence_end")))
        assert (row["vj_in_frame"], row["stop_codon"], row["productive"]) == (call.vj_in_frame, call.stop_codon,
                                                                               call.productive)
        assert row["productive_prediction"] == call.productive
        if k < 4:
            start, end = record["junction_start"], record["junction_end"]
            assert (row["junction_start"], row["junction_end"]) == (start, end)
            assert row["junction"] == record["junction"].upper() == row["sequence"][start:end]
            assert row["junction_aa"] == translate_nt(row["junction"]) and row["junction_length"] == end - start
            assert (row["cdr3"], row["cdr3_start"], row["cdr3_end"]) == (row["junction"][3:-3], start + 3, end - 3)
            assert row["vj_in_frame"] is not None and row["stop_codon"] is not None
        else:
            assert all(row[key] is None for key in JUNCTION_KEYS) and row["productive"] is None
            assert row["vj_in_frame"] is None and row["stop_codon"] is None


@pytest.mark.integration
def test_a_reverse_complemented_read_takes_its_junction_from_anchors_in_the_forward_frame(golden):
    """Reads presented reverse-complemented: the model calls orientation 1 and gives its anchors in the forward frame
    it corrects the read to. The row's sequence is the forward read and its junction GenAIRR's; the AIRR row written
    for it carries the presented read with rev_comp T, its CDR3 positions on the reverse complement of that read (the
    forward read), as AIRR places them."""
    import GenAIRR.data as gd
    from alignair.api import predict_sequences
    from alignair.io.airr import _build_row
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    model, reference = golden
    complement = str.maketrans("ACGTN", "TGCAN")
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    records = [r for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[1], 33), 8)
               if r["junction_start"] is not None][:3]
    forward = [r["sequence"].upper() for r in records]
    presented = [read.translate(complement)[::-1] for read in forward]
    anchors = {_encoded(read): (r["junction_start"], r["junction_end"] - 3) for read, r in zip(presented, records)}
    stub = AnchorStub(model, anchors, orientations={_encoded(read): 1 for read in presented})
    rows = predict_sequences(stub, reference, presented, device="cpu", batch_size=4)
    for row, record, read, shown in zip(rows, records, forward, presented):
        start, end = record["junction_start"], record["junction_end"]
        assert row["orientation_id"] == 1 and row["sequence"] == read and row["input_sequence"] == shown
        assert (row["junction_start"], row["junction_end"]) == (start, end)
        assert row["junction"] == record["junction"].upper() == read[start:end]
        written = _build_row("r", shown, row, "IGH")
        assert written["rev_comp"] == "T" and written["sequence"] == shown
        on_reverse_complement = written["sequence"].translate(complement)[::-1]
        assert on_reverse_complement[written["cdr3_start"] - 1 - 3:written["cdr3_end"] + 3] == row["junction"]


@pytest.mark.integration
def test_an_anchor_models_light_records_carry_the_derived_productivity(golden):
    """Without the AIRR assembly, a record still carries its anchors and the productivity they give, never a head's
    call (there is none)."""
    import GenAIRR.data as gd
    from alignair.api import predict_sequences
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    model, reference = golden
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    records = [r for r in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[1], 32), 8)
               if r["junction_start"] is not None][:3]
    reads = [r["sequence"].upper() for r in records]
    anchors = {_encoded(read): (r["junction_start"], r["junction_end"] - 3) for read, r in zip(reads, records)}
    rows = predict_sequences(AnchorStub(model, anchors), reference, reads, device="cpu", airr=False)
    for row, read in zip(rows, reads):
        v_anchor, j_anchor = anchors[_encoded(read)]
        call = derive_junction(row["sequence"], v_anchor, j_anchor,
                               coding_span=(row.get("v_sequence_start"), row.get("j_sequence_end")))
        assert (row["v_anchor"], row["j_anchor"], row["productive"]) == (v_anchor, j_anchor, call.productive)
        assert "junction" not in row and "productive_prediction" not in row
