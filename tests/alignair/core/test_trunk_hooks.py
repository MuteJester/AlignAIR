"""Product trunk hooks: conv-v1 is unchanged, and an architecture's trunk is computed once and shared."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import GenAIRR.data as gd

from alignair.core import AlignAIR, ModelComponents, build_model_config, prepare_model, register_architecture
from alignair.core import components
from alignair.core.gene_branch import GeneBranch
from alignair.errors import ConfigurationError
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component

GOLDEN = Path(__file__).parent / "fixtures" / "conv_v1_forward.pt"


def golden_model_and_batch():
    torch.manual_seed(0)
    model = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)).build_model().eval()
    length = model.cfg.max_seq_length
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randint(1, 6, (4, length), generator=generator)
    for row, valid in enumerate((length, 300, 200, 120)):
        tokens[row, valid:] = 0
    return model, {"tokenized_sequence": tokens}


def test_conv_v1_matches_its_golden_forward():
    model, batch = golden_model_and_batch()
    golden = torch.load(GOLDEN)
    assert list(model.state_dict()) == golden["state_keys"]
    with torch.no_grad():
        out = model(batch)
    assert set(out) == set(golden["outputs"])
    for key, value in golden["outputs"].items():
        assert torch.equal(out[key], value), key


class CountingTrunk(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.embed_dim, 3)
        self.calls = 0
        self.last = None

    def forward(self, emb, mask):
        self.calls += 1
        self.last = self.proj(emb) * mask.unsqueeze(-1)
        return self.last


class SeenBranch(GeneBranch):
    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        self.seen = {}

    def segment(self, emb, mask=None, end_mask=None, *, features=None):
        self.seen["segment"] = features
        return super().segment(emb, mask, end_mask)

    def classify(self, emb, s_exp, e_exp, *, features=None):
        self.seen["classify"] = features
        return super().classify(emb, s_exp, e_exp)


class SeenMetaTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.inner = components._meta_tower(cfg)
        self.seen = None

    def forward(self, emb, *, features=None, mask=None):
        self.seen = features
        self.seen_mask = mask
        return self.inner(emb)


class ScoreHead(nn.Module):
    def forward(self, features, mask):
        return features.sum(-1) * mask


@pytest.fixture
def architecture(monkeypatch):
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))

    def register(**factories):
        name = "test-trunk-hooks-v1"
        register_architecture(name, replace(ModelComponents(), **factories))
        return name
    return register


def config(**options):
    return build_model_config(3, 2, max_seq_length=128, embed_dim=4, filters=4, block_out=8, **options)


def trunk_architecture(architecture, **extra):
    return architecture(trunk=CountingTrunk, gene_branch=SeenBranch, meta_tower=SeenMetaTower, **extra)


def run(model):
    tokens = torch.zeros(2, 128, dtype=torch.long)
    tokens[:, :40] = torch.randint(1, 6, (2, 40))
    with torch.no_grad():
        return model({"tokenized_sequence": tokens, "orientation": torch.zeros(2, dtype=torch.long)})


def test_a_trunk_runs_once_and_reaches_branches_and_meta_tower(architecture):
    model = AlignAIR(config(architecture=trunk_architecture(architecture))).eval()
    run(model)
    assert model.trunk.calls == 1
    features = model.trunk.last
    assert model.meta_tower.seen is features
    for branch in model.branches.values():
        assert branch.seen["segment"] is features and branch.seen["classify"] is features
    assert model.component_modules()["trunk"] is model.trunk


def test_with_a_trunk_the_meta_tower_is_given_the_read_mask(architecture):
    model = AlignAIR(config(architecture=trunk_architecture(architecture))).eval()
    run(model)
    assert torch.equal(model.meta_tower.seen_mask, torch.arange(128).expand(2, -1) < 40)


def test_trunk_heads_write_their_outputs(architecture):
    name = trunk_architecture(architecture, trunk_heads=lambda cfg: {"score": ScoreHead()})
    model = AlignAIR(config(architecture=name)).eval()
    out = run(model)
    assert torch.equal(out["score"], model.trunk.last.sum(-1) * (torch.arange(128) < 40))
    assert model.component_modules()["trunk_heads.score"] is model.trunk_heads["score"]


def test_trunk_heads_without_a_trunk_are_refused(architecture):
    name = architecture(trunk_heads=lambda cfg: {"score": ScoreHead()})
    with pytest.raises(ConfigurationError) as caught:
        AlignAIR(config(architecture=name))
    assert caught.value.context["component"] == "trunk_heads"


def test_a_model_with_a_trunk_round_trips_through_its_state_dict(architecture):
    name = trunk_architecture(architecture, trunk_heads=lambda cfg: {"score": ScoreHead()})
    cfg = config(architecture=name)
    model = AlignAIR(cfg).eval()
    state = model.state_dict()
    assert any(key.startswith("trunk.") for key in state)
    restored = AlignAIR.from_state_dict(cfg, state)
    assert list(restored.state_dict()) == list(state)
    torch.manual_seed(3)
    first = run(model)
    torch.manual_seed(3)
    second = run(restored)
    assert torch.equal(first["score"], second["score"])
