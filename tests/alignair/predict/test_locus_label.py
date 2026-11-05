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


def test_a_multi_locus_reference_labels_each_row_with_its_called_vs_locus():
    """IGK + IGL: a row's locus is the locus of its V call (never IGH), and blank without a V call."""
    import GenAIRR.data as gd
    import torch
    from alignair.api import predict_sequences
    from alignair.core import prepare_model
    from alignair.reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)
    torch.manual_seed(0)
    model = prepare_model(reference, embed_dim=4, trunk_channels=8).build_model().eval()
    reads = [reference.gene("V").sequences[i] + reference.gene("J").sequences[0] for i in (0, 5, -1)]
    rows = predict_sequences(model, reference, reads, device="cpu")
    of = {locus: set(np.asarray(reference.gene("V").names)[reference.locus_mask(locus, "V")]) for locus in ("IGK", "IGL")}
    for row in rows:
        call = row["v_call"].split(",")[0]
        assert row["locus"] == ("" if not call else next(locus for locus, names in of.items() if call in names))
