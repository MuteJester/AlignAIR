"""Each read's prediction is typed (``predict.results``): a gate sets a gene's status and keeps its evidence, and a
read's light record (``light_record``) reports it with the keys in their order: a read without a V and with a short D
reports neither call, keeps the D span, and is labelled V absent and D short."""
import numpy as np
import pytest

from alignair.core.d_presence import SHORT
from alignair.predict.results import Anchors, GeneStatus, light_record, read_results
from alignair.predict.state import GeneCall, GermlineAlignment, Predictions

pytestmark = pytest.mark.component

SEQ = "ACGT" * 30


def _read(*, d_presence, no_v):
    calls = {"v": [GeneCall(("V1",), (1.0,))], "d": [GeneCall(("D1",), (0.7,))], "j": [GeneCall(("J1",), (0.9,))]}
    alignments = {"v": [GermlineAlignment("V1", 0, 40, 0, 40, "40M")],
                  "d": [GermlineAlignment("D1", 50, 60, 2, 12, "10M")],
                  "j": [GermlineAlignment("J1", 70, 100, 0, 30, "30M")]}
    predictions = Predictions(allele={}, start={}, end={}, mutation_rate=np.array([0.05]),
                              indel_count=np.array([0.0]), orientation=np.array([0]),
                              anchors={"v": np.array([10]), "j": np.array([80])}, d_presence=np.array([d_presence]))
    (read,) = read_results([SEQ], [SEQ], predictions, calls, alignments, np.array([False]), [False], [no_v],
                           [("V1",)], ["IGH"])
    return read


def test_a_gate_sets_a_status_and_keeps_the_evidence():
    read = _read(d_presence=SHORT, no_v=True)
    assert read.genes["v"].status is GeneStatus.ABSENT and read.genes["v"].alignment is not None
    assert read.genes["v"].placed is None and not read.v_present
    assert read.genes["d"].status is GeneStatus.ABSTAINED and read.genes["d"].placed is not None
    assert read.genes["j"].status is GeneStatus.CALLED and read.anchors == Anchors(10, 80)


def test_a_read_without_a_v_and_with_a_short_d_reports_neither_call():
    record = light_record(_read(d_presence=SHORT, no_v=True))
    assert list(record) == ["sequence", "orientation_id", "mutation_rate", "indel_count", "productive", "v_anchor",
                            "j_anchor", "input_sequence", "segmentation_low_quality", "locus", "v_call", "v_calls",
                            "v_likelihoods", "d_call", "d_calls", "d_likelihoods", "d_sequence_start",
                            "d_sequence_end", "d_germline_start", "d_germline_end", "d_cigar", "j_call", "j_calls",
                            "j_likelihoods", "j_sequence_start", "j_sequence_end", "j_germline_start",
                            "j_germline_end", "j_cigar", "d_presence", "v_presence"]
    assert (record["v_call"], record["v_anchor"], record["d_call"], record["d_presence"], record["v_presence"],
            record["productive"]) == ("", None, "", "short", "absent", None)


def test_a_called_read_reports_its_calls_and_placements():
    record = light_record(_read(d_presence=2, no_v=False))
    assert (record["v_call"], record["v_calls"], record["d_call"], record["j_likelihoods"]) == (
        "V1", ["V1"], "D1", [0.9])
    assert record["v_sequence_end"] == 40 and record["d_presence"] == "callable" and "v_presence" not in record


def test_a_multi_locus_read_without_a_v_has_a_blank_locus():
    import GenAIRR.data as gd
    from alignair.predict.stages import record_loci
    from alignair.reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)
    igk = reference.gene("V").names[int(np.flatnonzero(reference.locus_mask("IGK", "V"))[0])]
    alignment = GermlineAlignment(igk, 0, 10, 0, 10, "10M")
    assert record_loci([alignment, alignment], reference, [True, False], ("IGK", "IGL")) == ["", "IGK"]
