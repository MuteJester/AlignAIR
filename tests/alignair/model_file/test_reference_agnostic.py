"""Loading with another reference: the model (its alleles an input) binds the reference it is given, alleles it never
trained on included."""
import GenAIRR.data as gd
import pytest

from alignair.core import prepare_model
from torch import nn

from alignair import Aligner
from alignair import model_file as mf
from alignair.core import AlignAIR, ModelComponents, register_architecture
from alignair.core import components
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
    register_architecture("test-bank-v1", ModelComponents(core=BankCore, objective=lambda cfg: None))

    def save(architecture):
        cfg = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=576,
                            architecture=architecture).config
        path = tmp_path / f"{architecture}.alignair"
        mf.save_model(str(path), AlignAIR(cfg), ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                      dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
        return str(path)
    return save


def _another_reference():
    """The first ten V alleles of HUMAN_IGH_OGRDB and a novel one (the first with its first base changed)."""
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    subset = reference.subset({"V": reference.gene("V").names[:10]})
    v = subset.gene("V")
    novel = ("A" if v.sequences[0][0] != "A" else "C") + v.sequences[0][1:]
    return ReferenceSet.from_alleles(
        {"V": {**dict(zip(v.names, v.sequences)), "IGHV-novel*01": novel},
         "D": dict(zip(subset.gene("D").names, subset.gene("D").sequences)),
         "J": dict(zip(subset.gene("J").names, subset.gene("J").sequences))})


def test_a_model_loads_with_another_reference(saved):
    path = saved("test-bank-v1")
    other = _another_reference()
    loaded = mf.load_model(path, reference=other)
    assert loaded.reference.gene("V").names == other.gene("V").names
    assert loaded.model.core.reference is loaded.reference
    assert len(loaded.reference.gene("V")) == 11
    aligner = Aligner.from_pretrained(path, reference=other, device="cpu")
    assert aligner.reference.gene("V").names == other.gene("V").names
    assert aligner.model.core.reference is aligner.reference


def test_a_reference_bound_at_load_is_annotated_against_the_models_own(saved):
    """A Python user binding a reference gets what the CLI's --reference gets: the known alleles keep the model's
    anchors, a SNP novel allele takes its nearest allele's gaps and anchor."""
    from alignair.reference import AnchorEvidence, anchor_report
    path = saved("test-bank-v1")
    other = _another_reference()
    novel = other.gene("V").sequences[other.gene("V").index["IGHV-novel*01"]]
    trained = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB).gene("V")
    for _, reference in (("model_file", mf.load_model(path, reference=other).reference),
                         ("api", Aligner.from_pretrained(path, reference=other, device="cpu").reference)):
        v = reference.gene("V")
        assert v.anchors["IGHV-novel*01"] == 288 and v.gapped["IGHV-novel*01"].replace(".", "") == novel
        assert v.info["IGHV-novel*01"].anchor_evidence == AnchorEvidence(
            "nearest_known_allele", "inferred", "1 of 301 bases differ", "IGHVF1-G1*01")
        assert {n: v.anchors[n] for n in v.names[:10]} == {n: trained.anchors[n] for n in v.names[:10]}
        assert anchor_report(reference).complete()
