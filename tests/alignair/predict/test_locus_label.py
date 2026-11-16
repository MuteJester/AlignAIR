"""A record is labelled with the reference's locus when the reference has one (``stages.record_loci``)."""
from types import SimpleNamespace

import numpy as np
import pytest

from alignair.predict.results import light_record, read_results
from alignair.predict.stages import record_loci
from alignair.predict.state import GeneCall, GermlineAlignment, Predictions
from tests.support.read_results import prediction_state

pytestmark = pytest.mark.component


def _record(reference_loci):
    calls = {"v": [GeneCall(("IGKV1",), (1.0,))], "j": [GeneCall(("IGKJ1",), (1.0,))]}
    alignments = {"v": [GermlineAlignment("IGKV1", 0, 10, 0, 10, "10M")],
                  "j": [GermlineAlignment("IGKJ1", 20, 30, 0, 10, "10M")]}
    preds = Predictions(allele={}, start={}, end={}, mutation_rate=np.array([0.0]), indel_count=np.array([0.0]),
                        orientation=np.array([0]))
    no_schema = SimpleNamespace(locus_names=tuple)                       # a reference without a locus schema
    loci = record_loci(alignments["v"], no_schema, None, reference_loci)
    (read,) = read_results(prediction_state(["ACGT"], preds, calls, alignments), loci)
    return light_record(read)


def test_a_one_locus_reference_labels_the_record():
    assert _record(("IGK",))["locus"] == "IGK"


@pytest.mark.parametrize("reference_loci", [None, ("IGH", "IGK")])
def test_no_locus_label_without_a_one_locus_reference(reference_loci):
    assert "locus" not in _record(reference_loci)


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



def test_a_rows_locus_is_its_reported_v_alleles_not_the_models_top_name():
    """IGK + IGL: the locus of a row comes from the allele its V is aligned to (the reported V call, which may be an
    allele the read matches better than the model's top name, ``predict.ties.better_v_alleles``)."""
    import GenAIRR.data as gd
    from alignair.predict.stages import called_loci as _called_loci
    from alignair.reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)
    v = reference.gene("V")
    igk = v.names[int(np.flatnonzero(reference.locus_mask("IGK", "V"))[0])]
    igl = v.names[int(np.flatnonzero(reference.locus_mask("IGL", "V"))[0])]
    alignments = [GermlineAlignment(igl, 0, 10, 0, 10, "10M"), GermlineAlignment(igk, 0, 10, 0, 10, "10M"), None]
    assert _called_loci(alignments, reference, np.array([False, True, False])) == ["IGL", "", ""]


def test_every_rows_v_d_and_j_are_of_its_locus_and_a_light_row_has_no_d():
    """IGH + IGK, an untrained tiny chip: whatever the model calls, a row's V, D and J alleles are of one locus (the
    row's), and an IGK row has no D call and no D placement (``refine.locus``)."""
    import GenAIRR.data as gd
    import torch
    from alignair.api import predict_sequences
    from alignair.core import prepare_model
    from alignair.reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGK_OGRDB)
    torch.manual_seed(0)
    model = prepare_model(reference, embed_dim=4, trunk_channels=8).build_model().eval()

    def of(locus, gene):
        return np.asarray(reference.gene(gene).names)[reference.locus_mask(locus, gene)]

    def germline(locus, gene, k):
        alleles = reference.gene(gene)
        return alleles.sequences[alleles.index[of(locus, gene)[k]]]

    reads = ([germline("IGH", "V", k) + germline("IGH", "D", k) + germline("IGH", "J", 0) for k in (0, 7, 20)]
             + [germline("IGK", "V", k) + germline("IGK", "J", 0) for k in (0, 7, 20)])
    rows = predict_sequences(model, reference, reads, device="cpu")
    assert {row["locus"] for row in rows} <= {"IGH", "IGK", ""}
    for row in rows:
        if not row["locus"]:
            continue
        for gene in "vdj":
            names = set(of(row["locus"], gene.upper()))
            assert all(name in names for name in filter(None, row[f"{gene}_call"].split(","))), (row["locus"], gene)
        if row["locus"] == "IGK":
            assert not row["d_call"] and row.get("d_sequence_start") in (None, "")


def test_a_rows_locus_is_the_reads_locus_when_its_v_allele_is_shared():
    """TRA + TRD: a read's label is its locus (``read_locus``, the network's), so a TRD read whose V is a TRAV/DV
    allele (listed by both loci) is labelled TRD."""
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRA_IMGT, gd.HUMAN_TCRD_IMGT)
    aln = GermlineAlignment("TRAV29/DV5*01", 0, 10, 0, 10, "10M")
    trd = reference.locus_names().index("TRD")
    assert record_loci([aln, None], reference, [False, True], None, read_locus=np.array([trd, trd])) == ["TRD", ""]
    assert record_loci([aln], reference, [False], None) == ["TRA"]            # without it: the allele's first locus
