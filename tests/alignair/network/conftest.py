"""A toy chip: a 21-base read (V [0, 8), N1, D [10, 14), N2, J [16, 21)) against a two-allele-per-gene reference, with
chip-v1 (a 128-channel trunk) built and bound to it."""
import pytest
import torch

from alignair.core import prepare_model
from alignair.data import NucleotideBatchAdapter
from alignair.reference.reference_set import GeneReference, ReferenceSet


SEQUENCE = "ACGTACGT" + "AT" + "GGCC" + "CT" + "TGGAA"
SPANS = {"v": (0, 8, 0, 8), "d": (10, 14, 1, 5), "j": (16, 21, 2, 7)}
OPTIONS = dict(architecture="chip-v1", trunk_channels=128, max_seq_length=64)


def toy_reference():
    v = GeneReference(["V*01", "V*02"], ["ACGTACGTAC", "TTTTTTTTTT"], {"V*01": 0, "V*02": 1},
                      anchors={"V*01": 4, "V*02": 4}, gapped={"V*01": "ACGT..ACGTAC", "V*02": "TTTT..TTTTTT"})
    d = GeneReference(["D*01", "D*02"], ["GGGCCC", "TTTAAA"], {"D*01": 0, "D*02": 1})
    j = GeneReference(["J*01", "J*02"], ["TTTGGAA", "TTAGGAA"], {"J*01": 0, "J*02": 1},
                      anchors={"J*01": 3, "J*02": 3})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def with_a_novel_v():
    """The toy reference plus V*03, a novel allele (V*01 with its last base changed), gapped and anchored like V*01."""
    toy = toy_reference()
    v = GeneReference(["V*01", "V*02", "V*03"], ["ACGTACGTAC", "TTTTTTTTTT", "ACGTACGTAA"],
                      {"V*01": 0, "V*02": 1, "V*03": 2}, anchors={"V*01": 4, "V*02": 4, "V*03": 4},
                      gapped={"V*01": "ACGT..ACGTAC", "V*02": "TTTT..TTTTTT", "V*03": "ACGT..ACGTAA"})
    return ReferenceSet({"V": v, "D": toy.gene("D"), "J": toy.gene("J")}, has_d=True)


def toy_record():
    record = {"sequence": SEQUENCE, "sequence_id": "chip-read", "productive": None, "mutation_rate": 0.,
              "n_indels": 0, "n_quality_errors": 0, "v_call": "V*01", "d_call": "D*01", "j_call": "J*01"}
    for gene, (start, end, gs, ge) in SPANS.items():
        record.update({f"{gene}_sequence_start": start, f"{gene}_sequence_end": end, f"{gene}_germline_start": gs,
                       f"{gene}_germline_end": ge, f"{gene}_cigar": f"{end - start}M"})
    return record


@pytest.fixture
def chip():
    """(model, batch, setup): the toy chip and a two-read batch of its training inputs."""
    torch.manual_seed(0)
    reference = toy_reference()
    setup = prepare_model(reference, **OPTIONS)
    model = setup.build_model()
    batch = NucleotideBatchAdapter(reference, setup.config, augment_orientation=False)([toy_record(), toy_record()])
    return model, batch, setup
