"""A genotype is the model's bank: ``Aligner.with_genotype`` binds a copy of the model to the genotype's subset of the
reference once (no post-hoc mask), and leaves the Aligner it came from bound to the whole reference."""
import GenAIRR.data as gd
import pytest
from torch import nn

from alignair import Aligner
from alignair.core import AlignAIR, ModelComponents, components, prepare_model, register_architecture
from alignair.errors import InputError
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component


class BankCore(nn.Module):
    """A core that keeps the reference it is bound to."""

    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.embed_dim, 1)
        self.reference = None

    def bind_reference(self, reference):
        self.reference = reference

    def forward(self, emb, mask, end_mask, *, tokens, batch):
        return {}


@pytest.fixture
def aligner(monkeypatch):
    """An Aligner over a model whose core records the reference it is bound to."""
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))
    register_architecture("test-bank-v1", ModelComponents(core=BankCore, objective=lambda cfg: None))
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    model = AlignAIR(prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=576,
                                   architecture="test-bank-v1").config)
    model.bind_reference(reference)
    return Aligner.from_model(model, reference, device="cpu")


def test_a_genotype_is_the_models_bank(aligner):
    reference = aligner.reference
    genotype = {"v": set(reference.gene("V").names[:3]), "j": set(reference.gene("J").names[:2])}
    donor = aligner.with_genotype(genotype)
    assert donor.model.core.reference is donor.reference
    assert set(donor.reference.gene("V").names) == genotype["v"]
    assert set(donor.reference.gene("J").names) == genotype["j"]
    assert donor.reference.gene("D").names == reference.gene("D").names         # a gene it does not name: all
    assert donor.model is not aligner.model and aligner.model.core.reference is reference


def test_a_genotype_naming_an_allele_the_reference_lacks_is_refused(aligner):
    reference = aligner.reference
    with pytest.raises(InputError, match="IGHV-missing"):
        aligner.with_genotype({"v": {"IGHV-missing*01"}})
    assert aligner.model.core.reference is reference
