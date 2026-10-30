"""The AIRR rows of a model with anchor heads take their junction, frame, stop codon and productivity from the decoded
anchors (``predict.junction.derive_junction``, over the row's own V start and J end); a model without them, conv-v1,
gives the rows it gave before the anchors existed.

A decoded anchor is the argmax of its logits over the read's positions and the null slot L; the null slot means the
anchor is not observed (None)."""
import itertools
import json
from dataclasses import replace

import pytest
import torch

from alignair.data.projection import translate_nt
from alignair.data.tokenizer import TOKEN_DICT
from alignair.predict.decode import decode_anchors
from alignair.predict.junction import derive_junction

from tests.alignair.predict.golden_rows import FIXTURE, golden_model, predict_rows

JUNCTION_KEYS = ("junction", "junction_aa", "junction_length", "junction_aa_length", "junction_start", "junction_end",
                 "cdr3", "cdr3_aa", "cdr3_start", "cdr3_end")


@pytest.fixture(scope="module")
def golden():
    return golden_model()


@pytest.mark.component
def test_anchors_decode_by_argmax_over_the_read_and_the_null_slot():
    """Pad slots never win, the null slot (L) can; a tie takes the first slot; the result is a long position, L
    for the null slot."""
    read = torch.zeros(4, 6, dtype=torch.bool)
    read[:, :4] = True
    logits = torch.full((4, 7), -5.0)
    logits[0, 2] = 3.0                       # a read position
    logits[1, 5] = 9.0                       # a pad: never decoded
    logits[1, 1] = 1.0
    logits[2, 6] = 4.0                       # the null slot
    logits[3, [1, 3]] = 2.0                  # a tie
    assert decode_anchors(logits, read).tolist() == [2, 1, 6, 1]


@pytest.mark.component
def test_anchors_refuse_a_non_finite_logit_on_an_allowed_slot():
    from alignair.errors import NumericalError
    read = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(NumericalError):
        decode_anchors(torch.tensor([[0.0, float("nan"), 0.0, 0.0]]), read)


@pytest.mark.integration
def test_conv_v1_rows_are_unchanged(golden):
    """The golden conv-v1's rows on the fixture's reads (full length, with and without a V indel, J-anchored
    amplicons, a read longer than the window) are the rows recorded before the anchor heads existed, which already
    had their CDR3 positions in the read: the junction keeps its two paths, productive its derivation from the
    alignment, and productive_prediction the learned head's call."""
    fixture = json.loads(FIXTURE.read_text())
    rows = predict_rows(*golden, fixture["reads"])
    assert rows == fixture["rows"]
    assert all("v_anchor" not in row for row in rows)


class AnchorStub(torch.nn.Module):
    """The shaped golden conv-v1 with anchor heads in place of its productive head: its outputs less ``productive``,
    the forward orientation for every read, and V and J anchor logits peaked at the positions ``anchors`` gives each
    read (keyed by its encoded bases; None: the null slot)."""

    def __init__(self, inner, anchors: dict):
        super().__init__()
        self.inner, self.anchors = inner, anchors
        self.cfg = replace(inner.cfg, anchor_heads=True)

    def forward(self, batch):
        out = self.inner(batch)
        del out["productive"]
        tokens = batch["tokenized_sequence"]
        rows, length = tokens.shape
        out["orientation"] = torch.zeros(rows, dtype=torch.long)
        out["orientation_logits"] = torch.nn.functional.one_hot(out["orientation"], 4).float() * 10
        for k, key in enumerate(("v_anchor_logits", "j_anchor_logits")):
            logits = torch.zeros(rows, length + 1)
            for row in range(rows):
                position = self.anchors[tuple(tokens[row][tokens[row] != 0].tolist())][k]
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
