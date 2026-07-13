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
    assert "chain_type_id" not in recs[0]                # no multi-chain chain_type_id...


def test_to_records_single_locus_labels_without_chain_type():
    """Single-chain models (no chain_type head) still label the record's one locus (no silent IGH)."""
    calls, alns, preds = _preds(chain_type=None)
    recs = _to_records(["ACGT"], calls, alns, ("v", "j"), preds, chain_types=("IGK",))
    assert recs[0]["locus"] == "IGK"


# --- P0-6: per-read locus masking makes cross-locus calls impossible by construction ---------------

def test_locus_allowed_restricts_each_read_to_its_locus():
    import GenAIRR.data as gd
    from alignair.predict.pipeline import _locus_allowed
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)   # loci = (IGK, IGL)
    chain_type = np.array([0, 1])                        # read 0 -> IGK, read 1 -> IGL
    allowed = _locus_allowed(chain_type, ref, ("v", "j"))
    v_names = list(ref.gene("V").names)
    igk_idx = [i for i, n in enumerate(v_names) if n.upper().startswith("IGKV")]
    igl_idx = [i for i, n in enumerate(v_names) if n.upper().startswith("IGLV")]
    assert allowed["v"].shape == (2, len(v_names))
    assert allowed["v"][0][igk_idx].all() and not allowed["v"][0][igl_idx].any()   # read 0 = IGK only
    assert allowed["v"][1][igl_idx].all() and not allowed["v"][1][igk_idx].any()   # read 1 = IGL only


def test_locus_allowed_intersects_with_genotype():
    import GenAIRR.data as gd
    from alignair.predict.pipeline import _locus_allowed
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)
    v_names = list(ref.gene("V").names)
    geno = {"v": np.array([n == v_names[0] for n in v_names])}      # only the first IGK V allele
    allowed = _locus_allowed(np.array([0]), ref, ("v",), geno)
    assert allowed["v"][0].sum() == 1 and allowed["v"][0][0]        # locus ∩ genotype = {that allele}
