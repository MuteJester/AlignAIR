"""Source-to-model batch adaptation with independent, hand-written annotations."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from alignair.core import AlignAIR, AlignAIRConfig
from alignair.data import NucleotideBatchAdapter
from alignair.data.projection import translate_nt
from alignair.errors import ConfigurationError, TrainingDataError
from alignair.reference.reference_set import ReferenceSet, GeneReference
from alignair.tasks import build_tasks

pytestmark = pytest.mark.component


def _record():
    sequence = "CAGGTGCAGCTGTGCTGGGGCCAG"  # Q V Q L C W G Q
    record = {"sequence": sequence, "sequence_aa": "QVQLCWGQ", "sequence_id": "annotated-read",
              "junction_start": 0, "productive": True, "mutation_rate": .05, "n_indels": 0,
              "n_quality_errors": 0, "v_call": "V*01,V*02", "d_call": "D*01", "j_call": "J*01"}
    for gene, start, end in (("v", 0, 12), ("d", 12, 15), ("j", 15, 24)):
        record.update({gene+"_sequence_start": start, gene+"_sequence_end": end,
                       gene+"_germline_start": 0, gene+"_germline_end": end-start,
                       gene+"_cigar": str(end-start)+"M"})
    return record


def _reference():
    seq = _record()["sequence"]
    return ReferenceSet({gene: GeneReference([gene+"*01", gene+"*02"], [s, s],
                                            {gene+"*01": 0, gene+"*02": 1})
                         for gene, s in (("V", seq[:12]), ("D", seq[12:15]), ("J", seq[15:]))}, has_d=True)


def _config(length=128, **kwargs):
    return AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, has_d=True,
                          max_seq_length=length, **kwargs)


def test_prefix_window_clips_coordinates_and_masks_lost_genes_without_mutating_source():
    record = _record()
    original = deepcopy(record)
    batch = NucleotideBatchAdapter(_reference(), _config(14), augment_orientation=False)([record])
    assert batch.lengths.tolist() == [13]
    assert batch.targets["d_end"].item() == 13
    assert batch.targets["j_start"].item() == batch.targets["j_end"].item() == 0
    assert not batch.targets["j_supervise"].item()
    assert batch.targets["j_allele"].sum().item() == 0
    assert batch.targets["state_labels"].shape == (1, 14)              # padded to the model width
    assert batch.targets["d_supervise"].dtype == torch.bool
    assert record == original


def test_half_open_record_window_leaves_a_terminal_slot():
    batch = NucleotideBatchAdapter(_reference(), _config(14),
                                   augment_orientation=False)([_record()])
    assert batch.lengths.tolist() == [13]
    assert batch.inputs["tokenized_sequence"][0, 13] == 0
    assert batch.targets["d_end"].item() == 13


def test_strict_window_and_unknown_alleles_report_record_context():
    with pytest.raises(TrainingDataError) as caught:
        NucleotideBatchAdapter(_reference(), _config(14), overflow="error")([_record()])
    assert caught.value.context["record_index"] == 0
    assert caught.value.context["record_id"] == "annotated-read"
    record = _record()
    record["v_call"] = "UNKNOWN*01"
    with pytest.raises(TrainingDataError) as caught:
        NucleotideBatchAdapter(_reference(), _config())([record])
    assert caught.value.context["allele"] == "UNKNOWN*01"
    assert caught.value.context["gene"] == "V"


def test_orientation_is_repeatable_and_disabled_models_receive_forward_reads():
    adapter = NucleotideBatchAdapter(_reference(), _config())
    torch.manual_seed(14)
    first = adapter([_record(), _record()])
    torch.manual_seed(14)
    second = adapter([_record(), _record()])
    torch.testing.assert_close(first.inputs["tokenized_sequence"], second.inputs["tokenized_sequence"])
    torch.testing.assert_close(first.inputs["orientation"], first.targets["orientation"])
    unoriented = _config(use_orientation=False, bank_orientation=False)
    disabled = NucleotideBatchAdapter(_reference(), unoriented)([_record()])
    assert "orientation" not in disabled.inputs and "orientation" not in disabled.targets
    with pytest.raises(ConfigurationError, match="requires"):
        NucleotideBatchAdapter(_reference(), unoriented, augment_orientation=True)([_record()])


def test_record_adapter_to_shared_model_and_task_loss_backpropagates():
    cfg = _config(embed_dim=4, trunk_channels=8)
    adapter = NucleotideBatchAdapter(_reference(), cfg)
    batch = adapter([_record(), _record()])
    model = AlignAIR(cfg)
    model.bind_reference(_reference())
    tasks = build_tasks(cfg)
    loss, _ = tasks.loss(model(batch.inputs), batch.targets, tasks.make_weights())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embedding.token.weight.grad is not None


def test_record_adapters_reject_the_wrong_objective_and_malformed_records():
    cfg = _config(use_orientation=False, bank_orientation=False)
    adapter = NucleotideBatchAdapter
    with pytest.raises(TrainingDataError, match="mappings"):
        adapter(_reference(), cfg)([None])
    with pytest.raises(ConfigurationError, match="overflow"):
        adapter(_reference(), cfg, overflow="discard")([_record()])


def test_the_trainers_batch_builder_is_the_shared_adapter():
    from alignair.train.trainer import build_batch
    from alignair.data.adapters import build_nucleotide_batch

    assert build_batch is build_nucleotide_batch


def test_nucleotide_empty_read_policy_matches_the_model_configuration():
    record = _record()
    record["sequence"] = ""
    for gene in ("v", "d", "j"):
        record[gene+"_sequence_start"] = record[gene+"_sequence_end"] = 0
    batch = NucleotideBatchAdapter(_reference(), _config())([record])
    assert batch.lengths.tolist() == [0]
    for gene in ("v", "d", "j"):
        assert not batch.targets[gene+"_supervise"].item()
    assert (batch.targets["state_labels"] == -100).all()
    with pytest.raises(TrainingDataError, match="nonempty"):
        NucleotideBatchAdapter(_reference(), _config(reject_empty_sequences=True))([record])
