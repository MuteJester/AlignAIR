"""A model (half-open boundaries) trains on a read whose segment ends at the read's last base (end == n),
and predicts AIRR rows whose end is n."""
import math

import pytest
import torch

from alignair import Aligner
from alignair.core import AlignAIR, AlignAIRConfig
from alignair.data import NucleotideBatchAdapter
from alignair.reference.reference_set import GeneReference, ReferenceSet
from alignair.tasks import build_tasks
from alignair.train.engine import TrainingEngine

pytestmark = pytest.mark.integration

V_SEQ, D_SEQ, J_SEQ = "CAGGTGCAGCTG", "TGC", "TGGGGCCAG"
READ = V_SEQ + D_SEQ + J_SEQ          # 24 bases; J ends at the read's last base


def _record():
    record = {"sequence": READ, "sequence_id": "r0", "junction_start": 0, "productive": True, "mutation_rate": .0,
              "n_indels": 0, "n_quality_errors": 0, "v_call": "V*01", "d_call": "D*01", "j_call": "J*01"}
    for gene, start, end in (("v", 0, 12), ("d", 12, 15), ("j", 15, 24)):
        record.update({gene + "_sequence_start": start, gene + "_sequence_end": end, gene + "_germline_start": 0,
                       gene + "_germline_end": end - start, gene + "_cigar": f"{end - start}M"})
    return record


def _reference():
    return ReferenceSet({gene: GeneReference([gene + "*01"], [seq], {gene + "*01": 0})
                         for gene, seq in (("V", V_SEQ), ("D", D_SEQ), ("J", J_SEQ))}, has_d=True)


def _config(**kwargs):
    return AlignAIRConfig(has_d=True, max_seq_length=128,
                          embed_dim=4, trunk_channels=8, **kwargs)


def test_a_read_ending_at_its_last_base_gets_a_supervised_end_slot_and_a_training_step_runs():
    config = _config()
    batch = NucleotideBatchAdapter(_reference(), config, augment_orientation=False)([_record()])
    assert batch.targets["j_end"].item() == len(READ)
    model = AlignAIR(config)
    model.bind_reference(_reference())
    out = model(batch.inputs)
    assert bool(out["end_mask"][0, len(READ)]) and not bool(out["end_mask"][0, len(READ) + 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    total, parts, norm = TrainingEngine(model, optimizer, tasks=build_tasks(config)).step((batch.inputs, batch.targets))
    assert math.isfinite(total) and math.isfinite(norm)


def test_prediction_reports_an_end_at_the_read_length_when_the_logits_say_so():
    config = _config()
    model = AlignAIR(config).eval()
    model.bind_reference(_reference())

    def peak_at_the_slot_after_the_last_base(module, args, output):
        peaked = torch.full_like(output["j_end_logits"], -20.0)
        peaked[:, len(READ)] = 20.0
        return {**output, "j_end_logits": peaked}

    model.core.register_forward_hook(peak_at_the_slot_after_the_last_base)
    row = Aligner.from_model(model, _reference(), device="cpu").predict([READ], batch_size=1).records[0]
    assert row["j_sequence_end"] == len(READ)
    # the toy V germline (12 nt) holds no 15-mer, so its read holds no V (predict.v_presence)
    assert row["v_presence"] == "absent" and row["v_call"] == ""
    for gene in ("d", "j"):                              # the D presence gate may drop the toy read's D
        if f"{gene}_sequence_start" in row:
            assert 0 <= row[f"{gene}_sequence_start"] <= row[f"{gene}_sequence_end"] <= len(READ)
    # a read longer than the window is cropped to leave the terminal slot
    assert len(Aligner.from_model(model, _reference(), device="cpu").predict([READ * 8], batch_size=1).records) == 1


def test_a_logit_peak_on_slot_n_is_reported_as_the_end():
    from tests.support.boundaries import decoded_boundaries
    n = len(READ)
    logits = torch.full((1, 64), -5.0)
    logits[0, n] = 5.0
    outputs = {"j_start_logits": logits, "j_end_logits": logits, "start_mask": torch.arange(64)[None] < n,
               "end_mask": torch.arange(64)[None] <= n}
    start, end = decoded_boundaries(outputs, "j")
    assert end.tolist() == [n]
