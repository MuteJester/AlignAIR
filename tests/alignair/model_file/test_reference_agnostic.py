"""Loading with another reference: a reference-agnostic model (its alleles an input) binds the reference it is given,
alleles it never trained on included; a fixed-head model keeps the reference it trained on."""
import GenAIRR.data as gd
import pytest
from torch import nn

from alignair import api
from alignair import model_file as mf
from alignair.core import AlignAIR, ModelComponents, register_architecture
from alignair.core import components
from alignair.core.config import AlignAIRConfig
from alignair.errors import ModelError
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.integration


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
def saved(tmp_path, monkeypatch):
    """``save(architecture) -> path``: a fresh model of the architecture saved on HUMAN_IGH_OGRDB."""
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))
    register_architecture("test-bank-agnostic-v1", ModelComponents(core=BankCore, reference_agnostic=True))
    register_architecture("test-bank-fixed-v1", ModelComponents(core=BankCore))

    def save(architecture):
        cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576, architecture=architecture)
        path = tmp_path / f"{architecture}.alignair"
        mf.save_model(str(path), AlignAIR(cfg), dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
        return str(path)
    return save


def _another_reference():
    """The first ten V alleles of HUMAN_IGH_OGRDB and a novel one (the first with its first base changed)."""
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    subset = reference.subset({"V": reference.gene("V").names[:10]})
    v = subset.gene("V")
    novel = ("A" if v.sequences[0][0] != "A" else "C") + v.sequences[0][1:]
    return ReferenceSet.from_genotype(
        {"V": {**dict(zip(v.names, v.sequences)), "IGHV-novel*01": novel},
         "D": dict(zip(subset.gene("D").names, subset.gene("D").sequences)),
         "J": dict(zip(subset.gene("J").names, subset.gene("J").sequences))})


def test_a_reference_agnostic_model_loads_with_another_reference(saved):
    path = saved("test-bank-agnostic-v1")
    other = _another_reference()
    loaded = mf.load_model(path, reference=other)
    assert loaded.reference is other and loaded.model.core.reference is other
    assert loaded.config.v_allele_count == 11 and loaded.model.cfg.v_allele_count == 11
    model, reference = api.load_model(path, reference=other)
    assert reference is other and model.core.reference is other


def test_a_fixed_head_model_keeps_the_reference_it_trained_on(saved):
    path = saved("test-bank-fixed-v1")
    with pytest.raises(ModelError, match="reference does not match"):
        mf.load_model(path, reference=_another_reference())
    same = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    assert mf.load_model(path, reference=same).model.core.reference.gene("V").names == same.gene("V").names
