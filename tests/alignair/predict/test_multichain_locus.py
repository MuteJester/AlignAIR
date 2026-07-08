"""Multi-chain predict wiring: chain_type_logits -> chain_type index -> AIRR locus."""
import numpy as np

from alignair.predict.clean import clean
from alignair.predict.pipeline import _to_records
from alignair.predict.state import GeneCall, GermlineAlignment, Predictions


def test_clean_populates_chain_type_from_logits():
    # two reads, 3 chain types; argmax picks class 2 then class 0
    batch = {
        "v_allele": np.zeros((2, 4)), "j_allele": np.zeros((2, 4)),
        "v_start": np.zeros(2), "v_end": np.ones(2), "j_start": np.zeros(2), "j_end": np.ones(2),
        "mutation_rate": np.zeros(2), "indel_count": np.zeros(2), "productive": np.ones(2),
        "chain_type_logits": np.array([[0.1, 0.2, 5.0], [3.0, 0.1, 0.2]]),
    }
    preds = clean([batch], genes=("v", "j"))
    assert preds.chain_type is not None
    assert list(preds.chain_type) == [2, 0]


def test_clean_chain_type_none_when_absent():
    batch = {
        "v_allele": np.zeros((1, 4)), "j_allele": np.zeros((1, 4)),
        "v_start": np.zeros(1), "v_end": np.ones(1), "j_start": np.zeros(1), "j_end": np.ones(1),
        "mutation_rate": np.zeros(1), "indel_count": np.zeros(1), "productive": np.ones(1),
    }
    assert clean([batch], genes=("v", "j")).chain_type is None


def _preds(chain_type):
    call = {"v": [GeneCall(("IGHV1",), (1.0,))], "j": [GeneCall(("IGHJ1",), (1.0,))]}
    aln = {"v": [GermlineAlignment("IGHV1", 0, 10, 0, 10, "10M")],
           "j": [GermlineAlignment("IGHJ1", 20, 30, 0, 10, "10M")]}
    p = Predictions(allele={}, start={}, end={},
                    mutation_rate=np.array([0.0]), indel_count=np.array([0.0]),
                    productive=np.array([True]), orientation=np.array([0]),
                    chain_type=chain_type)
    return call, aln, p


def test_to_records_maps_chain_type_index_to_locus():
    calls, alns, preds = _preds(chain_type=np.array([1]))
    recs = _to_records(["ACGT"], calls, alns, ("v", "j"), preds,
                       chain_types=("IGH", "IGK", "IGL"))
    assert recs[0]["chain_type_id"] == 1
    assert recs[0]["locus"] == "IGK"


def test_to_records_no_locus_without_chain_type():
    calls, alns, preds = _preds(chain_type=None)
    recs = _to_records(["ACGT"], calls, alns, ("v", "j"), preds, chain_types=("IGH", "IGK"))
    assert "chain_type_id" not in recs[0] and "locus" not in recs[0]
