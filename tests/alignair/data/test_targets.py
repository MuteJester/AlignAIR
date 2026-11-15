"""Hand-written simulator contracts, independent of the live GenAIRR generator."""
from types import SimpleNamespace

import numpy as np
import pytest

from alignair.errors import TrainingDataError
from alignair.data.targets import allele_set, build_targets, observed_mutation_rate

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
    from alignair.data.adapters import NucleotideBatchAdapter
    from alignair.reference.reference_set import GeneReference, ReferenceSet
    from alignair.tasks.metrics import alignment_metrics

    reference = ReferenceSet({gene: GeneReference([gene + "*01"], ["ACGTACGTACGT"], {gene + "*01": 0})
                              for gene in ("V", "D", "J")}, has_d=True)
    cfg = AlignAIRConfig(max_seq_length=128, has_d=True)
    removed = _record()
    removed.update(j_sequence_start=12, j_sequence_end=12,
                   j_germline_start=None, j_germline_end=None, j_cigar="")
    _, targets = NucleotideBatchAdapter(reference, cfg, augment_orientation=False)([_record(), removed])
    assert targets["j_supervise"].tolist() == [True, False]
    assert targets["v_supervise"].tolist() == [True, True]
    outputs = {key: value.clone() for key, value in targets.items()}
    outputs["j_start"][1] = 90
    outputs["j_end"][1] = 100
    metrics = alignment_metrics(outputs, targets, cfg.genes).means()
    assert metrics["j_allele_top1"] == 1.0
    assert metrics["j_seg_mae"] == 0.0
    targets["j_supervise"].zero_()
    metrics = alignment_metrics(outputs, targets, cfg.genes).means()
    assert "j_allele_top1" not in metrics and "j_seg_mae" not in metrics


def _two_alleles():
    """A reference of two alleles per gene, ``<G>*01`` and ``<G>*02``."""
    return SimpleNamespace(gene=lambda gene: SimpleNamespace(
        index={gene + "*01": 0, gene + "*02": 1}, sequences=["ACGTACGTACGT", "ACGTACGTACGA"]))


def test_the_allele_set_is_the_call_and_its_recalled_alleles_the_reference_holds():
    record = {"v_call": "V*01", "recalled_v_call": "V*02, V*01,IGHV-NOT-IN-THE-BANK*01,"}
    assert allele_set(record, "v", _two_alleles()) == {"V*01", "V*02"}


def test_a_record_without_recalled_alleles_targets_its_call():
    for recalled in ({}, {"recalled_v_call": None}, {"recalled_v_call": ""}):
        assert allele_set({"v_call": "V*01", **recalled}, "v", _two_alleles()) == {"V*01"}


def test_the_recalled_alleles_join_the_records_targets():
    record = {**_record(), "recalled_v_call": "V*02", "recalled_j_call": "J*01"}
    calls = build_targets(record, _two_alleles(), True)["calls"]
    assert calls == {"V": {"V*01", "V*02"}, "D": {"D*01"}, "J": {"J*01"}}


def test_the_mutation_rate_target_is_observable_mutations_per_observed_germline_base():
    germline = {"v": (10, 290), "d": (2, 14), "j": (0, 40)}           # 280 + 12 + 40 = 332 germline bases
    record = {"n_observable_mutations": 33, "mutation_rate": 0.9}
    assert observed_mutation_rate(record, germline) == pytest.approx(33 / 332)


def test_a_record_from_before_genairr_2_4_keeps_its_mutation_rate():
    assert observed_mutation_rate({"mutation_rate": 0.07}, {"v": (0, 300)}) == 0.07


def test_a_record_with_no_observed_germline_has_a_rate_of_zero():
    assert observed_mutation_rate({"n_observable_mutations": 0, "mutation_rate": 0.1}, {}) == 0.0


def test_the_records_targets_carry_its_observed_mutation_rate():
    record = {**_record(), "n_observable_mutations": 3, "mutation_rate": 0.5}    # 3 of its 12 germline bases
    assert build_targets(record, _reference(), True)["mutation_rate"] == pytest.approx(0.25)
