"""A genotype is the model's bank: ``predict_sequences`` binds the genotype's subset of the reference for the call and
predicts against it (no post-hoc mask), then binds the reference back."""
import GenAIRR.data as gd
import pytest
from torch import nn

from alignair import api
from alignair.core import AlignAIR, ModelComponents, register_architecture
from alignair.core import components
from alignair.core.config import AlignAIRConfig
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
def make(monkeypatch):
    """``make() -> (model, reference)`` and the pipeline call ``predict_sequences`` makes, recorded."""
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))
    register_architecture("test-bank-v1", ModelComponents(core=BankCore, objective=lambda cfg: None))
    calls = []
    monkeypatch.setattr(api, "_predict", lambda model, sequences, reference, cfg, device: calls.append(
        {"bound": model.core.reference, "reference": reference}) or [])

    def build():
        reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
        model = AlignAIR(AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576,
                                                         architecture="test-bank-v1"))
        model.bind_reference(reference)
        return model, reference
    return build, calls


def test_a_genotype_is_the_models_bank(make):
    build, calls = make
    model, reference = build()
    genotype = {"v": set(reference.gene("V").names[:3]), "j": set(reference.gene("J").names[:2])}
    api.predict_sequences(model, reference, ["ACGT"], device="cpu", genotype=genotype)
    (call,) = calls
    assert call["bound"] is call["reference"]
    assert set(call["reference"].gene("V").names) == genotype["v"]
    assert set(call["reference"].gene("J").names) == genotype["j"]
    assert call["reference"].gene("D").names == reference.gene("D").names         # a gene it does not name: all
    assert model.core.reference is reference and model.cfg.v_allele_count == len(reference.gene("V"))


def test_a_genotype_naming_an_allele_the_reference_lacks_is_refused(make):
    build, calls = make
    model, reference = build()
    with pytest.raises(InputError, match="IGHV-missing"):
        api.predict_sequences(model, reference, ["ACGT"], device="cpu", genotype={"v": {"IGHV-missing*01"}})
    assert calls == [] and model.core.reference is reference

