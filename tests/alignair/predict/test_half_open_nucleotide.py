"""A nucleotide model with half-open boundaries trains on a read whose segment ends at the read's last base (end == n),
and predicts AIRR rows whose end is n."""
import math

import pytest
import torch

from alignair.api import predict_sequences
from alignair.core import AlignAIR, AlignAIRConfig
from alignair.data import NucleotideBatchAdapter
from alignair.reference.reference_set import GeneReference, ReferenceSet
from alignair.tasks import build_tasks
from alignair.train.engine import train_step

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
    return AlignAIRConfig(v_allele_count=1, j_allele_count=1, d_allele_count=1, has_d=True, max_seq_length=128,
                          embed_dim=4, filters=4, block_out=8, boundary_mode="half_open", **kwargs)


def test_a_read_ending_at_its_last_base_gets_a_supervised_end_slot_and_a_training_step_runs():
    config = _config()
    batch = NucleotideBatchAdapter(_reference(), config, augment_orientation=False)([_record()])
    assert batch.targets["j_end"].item() == len(READ)
    model = AlignAIR(config)
    out = model(batch.inputs)
    assert bool(out["end_mask"][0, len(READ)]) and not bool(out["end_mask"][0, len(READ) + 1])
    tasks = build_tasks(config)
    weights = tasks.make_weights()
    optimizer = torch.optim.AdamW([*model.parameters(), *weights.parameters()], lr=.001)
    total, parts, norm = train_step(model, batch.inputs, batch.targets, config, weights, optimizer, tasks=tasks)
    assert math.isfinite(total) and math.isfinite(norm)
    # the end slot n takes the supervised gradient
    out = model(batch.inputs)
    out["j_end_logits"].retain_grad()
    total, _ = tasks.loss(out, batch.targets, weights)
    total.backward()
    assert out["j_end_logits"].grad[0, len(READ)] < 0


def test_prediction_reports_an_end_at_the_read_length_when_the_logits_say_so():
    config = _config()
    model = AlignAIR(config).eval()

    def peak_at_the_slot_after_the_last_base(module, args, output):
        peaked = torch.full_like(output, -20.0)
        peaked[:, len(READ)] = 20.0
        return peaked

    model.branches["j"].end_head.register_forward_hook(peak_at_the_slot_after_the_last_base)
    row = predict_sequences(model, _reference(), [READ], device="cpu", batch_size=1)[0]
    assert row["j_sequence_end"] == len(READ)
    for gene in ("v", "d", "j"):
        assert 0 <= row[f"{gene}_sequence_start"] <= row[f"{gene}_sequence_end"] <= len(READ)
    # a read longer than the window is cropped to leave the terminal slot
    assert len(predict_sequences(model, _reference(), [READ * 8], device="cpu", batch_size=1)) == 1


def test_a_logit_peak_on_slot_n_is_reported_as_the_end():
    from alignair.predict.decode import decoded_boundaries
    n = len(READ)
    logits = torch.full((1, 64), -5.0)
    logits[0, n] = 5.0
    outputs = {"j_start_logits": logits, "j_end_logits": logits, "start_mask": torch.arange(64)[None] < n,
               "end_mask": torch.arange(64)[None] <= n}
    start, end = decoded_boundaries(outputs, "j")
    assert end.tolist() == [n]
