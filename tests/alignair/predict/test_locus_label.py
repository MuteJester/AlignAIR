"""A record is labelled with the reference's locus when the reference has one."""
import numpy as np
import pytest

from alignair.predict.pipeline import _to_records
from alignair.predict.state import GeneCall, GermlineAlignment, Predictions

pytestmark = pytest.mark.component


def _inputs():
    calls = {"v": [GeneCall(("IGKV1",), (1.0,))], "j": [GeneCall(("IGKJ1",), (1.0,))]}
    alignments = {"v": [GermlineAlignment("IGKV1", 0, 10, 0, 10, "10M")],
                  "j": [GermlineAlignment("IGKJ1", 20, 30, 0, 10, "10M")]}
    preds = Predictions(allele={}, start={}, end={}, mutation_rate=np.array([0.0]), indel_count=np.array([0.0]),
                        orientation=np.array([0]))
    return calls, alignments, preds


def test_a_one_locus_reference_labels_the_record():
    calls, alignments, preds = _inputs()
    assert _to_records(["ACGT"], calls, alignments, ("v", "j"), preds, chain_types=("IGK",))[0]["locus"] == "IGK"


@pytest.mark.parametrize("chain_types", [None, ("IGH", "IGK")])
def test_no_locus_label_without_a_one_locus_reference(chain_types):
    calls, alignments, preds = _inputs()
    assert "locus" not in _to_records(["ACGT"], calls, alignments, ("v", "j"), preds, chain_types=chain_types)[0]
