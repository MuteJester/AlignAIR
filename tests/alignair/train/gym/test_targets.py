"""Hand-written simulator contracts, independent of the live GenAIRR generator."""
from types import SimpleNamespace

import numpy as np
import pytest

from alignair.errors import TrainingDataError
from alignair.train.gym.targets import build_targets

pytestmark = pytest.mark.component


def _reference():
    return SimpleNamespace(gene=lambda gene: SimpleNamespace(
        index={gene + "*01": 0}, sequences=["ACGTACGTACGT"]))


def _record():
    return {
        "sequence": "ACGTACGTACGT", "sequence_id": "contract-read",
        "v_call": "V*01", "v_sequence_start": 0, "v_sequence_end": 4,
        "v_germline_start": 0, "v_germline_end": 4, "v_cigar": "4M",
        "d_call": "D*01", "d_sequence_start": 4, "d_sequence_end": 8,
        "d_germline_start": 0, "d_germline_end": 4, "d_cigar": "4M",
        "j_call": "J*01", "j_sequence_start": 8, "j_sequence_end": 12,
        "j_germline_start": 0, "j_germline_end": 4, "j_cigar": "4M",
        "n_quality_errors": 0, "mutation_rate": 0.0, "n_indels": 0,
        "productive": True,
    }


@pytest.mark.parametrize("gene", ["v", "d", "j"])
def test_removed_gene_has_no_alignment_supervision(gene):
    record = _record()
    record.update({f"{gene}_sequence_end": record[f"{gene}_sequence_start"],
                   f"{gene}_germline_start": None, f"{gene}_germline_end": None,
                   f"{gene}_cigar": ""})
    bundle = build_targets(record, _reference(), has_d=True)
    assert gene not in bundle["germline"]
    assert gene not in bundle["inseq"]
    assert np.isfinite(bundle["state_labels"]).all()
    assert len(bundle["tokens"]) == len(record["sequence"])


@pytest.mark.parametrize("gene", ["v", "d", "j"])
def test_visible_gene_requires_germline_coordinates(gene):
    record = _record()
    record[f"{gene}_germline_start"] = None
    record[f"{gene}_germline_end"] = None
    with pytest.raises(TrainingDataError) as caught:
        build_targets(record, _reference(), has_d=True)
    assert caught.value.code == "ALIGNAIR_TRAINING_DATA_INVALID"
    assert caught.value.context["gene"] == gene.upper()
    assert caught.value.context["record_id"] == "contract-read"


@pytest.mark.parametrize("start,end", [(-1, 4), (5, 4), (0, 13), (0, None), (None, 4),
                                      (True, 4), (1.5, 4)])
def test_invalid_query_span_is_rejected(start, end):
    record = _record()
    record.update(v_sequence_start=start, v_sequence_end=end)
    with pytest.raises(TrainingDataError):
        build_targets(record, _reference(), has_d=True)


@pytest.mark.parametrize("start,end", [(-1, 4), (5, 4), (0.5, 4), (0, True)])
def test_invalid_germline_span_is_rejected(start, end):
    record = _record()
    record.update(v_germline_start=start, v_germline_end=end)
    with pytest.raises(TrainingDataError) as caught:
        build_targets(record, _reference(), has_d=True)
    assert caught.value.context["gene"] == "V"
    assert caught.value.context["fields"] == ["v_germline_start", "v_germline_end"]


def test_batch_masks_removed_genes_in_training_and_metrics():
    from alignair.core.config import AlignAIRConfig
    from alignair.train.trainer import build_batch, eval_metrics
    import torch

    reference = _reference()
    reference.gene = lambda gene: SimpleNamespace(
        names=[gene+"*01"], index={gene+"*01": 0}, sequences=["ACGTACGTACGT"])
    cfg = AlignAIRConfig(max_seq_length=128, v_allele_count=1, j_allele_count=1,
                         d_allele_count=1, has_d=True)
    removed = _record()
    removed.update(j_sequence_start=12, j_sequence_end=12,
                   j_germline_start=None, j_germline_end=None, j_cigar="")
    _, targets = build_batch([_record(), removed], reference, cfg, augment_orientation=False)
    assert targets["j_supervise"].tolist() == [True, False]
    assert targets["v_supervise"].tolist() == [True, True]
    outputs = {key: value.clone() for key, value in targets.items()}
    outputs["j_start"][1] = 90
    outputs["j_end"][1] = 100
    metrics = eval_metrics(outputs, targets, cfg)
    assert metrics["j_allele_top1"] == 1.0
    assert metrics["j_seg_mae"] == 0.0
    targets["j_supervise"].zero_()
    metrics = eval_metrics(outputs, targets, cfg)
    assert "j_allele_top1" not in metrics and "j_seg_mae" not in metrics
