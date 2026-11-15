"""Architecture components: an architecture is its embedding, orientation head, network core and training objective."""
import pytest
import torch
from torch import nn

import GenAIRR.data as gd

from alignair.core import AlignAIR, AlignAIRConfig, ModelComponents, prepare_model, register_architecture
from alignair.core import components
from alignair.errors import ConfigurationError
from alignair.core.orientation import apply_orientation
from alignair.reference.reference_set import ReferenceSet
from alignair.tasks.objective import build_tasks

pytestmark = pytest.mark.component


class ToyCore(nn.Module):
    """Records what it is given and returns one output."""

    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.embed_dim, 1)
        self.seen = {}
        self.reference = None

    def bind_reference(self, reference):
        self.reference = reference

    def forward(self, emb, mask, end_mask, *, tokens, batch):
        self.seen = {"tokens": tokens, "mask": mask, "end_mask": end_mask, "batch": batch}
        return {"toy": self.proj(emb)[..., 0] * mask}


SENTINEL = object()


@pytest.fixture
def architecture(monkeypatch):
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))

    def register(**factories):
        name = "test-core-hooks-v1"
        register_architecture(name, ModelComponents(**{"core": ToyCore, "objective": lambda cfg: SENTINEL,
                                                       **factories}))
        return name
    return register


def config(**options):
    options.setdefault("bank_orientation", False)        # the toy core binds no bank
    return AlignAIRConfig(has_d=True, max_seq_length=128, embed_dim=4, **options)


def batch(orientation=0):
    tokens = torch.zeros(2, 128, dtype=torch.long)
    tokens[:, :30] = torch.randint(1, 5, (2, 30))
    return {"tokenized_sequence": tokens, "orientation": torch.full((2,), orientation, dtype=torch.long)}


def test_a_core_architecture_builds_only_embedding_orientation_and_core(architecture):
    model = AlignAIR(config(architecture=architecture(core=ToyCore))).eval()
    assert isinstance(model.core, ToyCore)
    assert set(dict(model.named_children())) == {"embedding", "orientation_head", "core"}
    with torch.no_grad():
        out = model(batch())
    assert {"toy", "start_mask", "end_mask", "orientation_logits"} <= set(out)
    assert model.component_modules()["core"] is model.core


def test_the_core_receives_orientation_corrected_tokens(architecture):
    model = AlignAIR(config(architecture=architecture(core=ToyCore))).eval()
    inputs = batch(orientation=1)
    with torch.no_grad():
        model(inputs)
    tokens = inputs["tokenized_sequence"]
    mask = tokens != 0
    assert torch.equal(model.core.seen["tokens"], apply_orientation(tokens, mask, inputs["orientation"]))
    assert model.core.seen["batch"] is inputs


def test_build_tasks_returns_the_architecture_objective(architecture):
    cfg = config(architecture=architecture(core=ToyCore, objective=lambda cfg: SENTINEL))
    assert build_tasks(cfg) is SENTINEL


def test_a_registered_core_is_the_models_core(architecture):
    model = AlignAIR(config(architecture=architecture(core=ToyCore)))
    assert isinstance(model.core, ToyCore)


def test_an_architecture_needs_a_core_and_an_objective():
    with pytest.raises(TypeError):
        ModelComponents(core=ToyCore)
    with pytest.raises(ConfigurationError, match="callable"):
        ModelComponents(core=ToyCore, objective="not callable")


def test_chip_v1_is_the_only_built_in_architecture():
    """In a fresh process (other tests register architectures of their own in this one)."""
    import os
    import subprocess
    import sys
    env = {key: value for key, value in os.environ.items() if key != "ALIGNAIR_ARCHITECTURE_MODULES"}
    code = "from alignair.core import components; print(sorted(components._ARCHITECTURES))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "['chip-v1']"
    with pytest.raises(ConfigurationError, match="unknown model architecture"):
        components.architecture_components("conv-v1")


def test_a_prepared_model_binds_its_reference_to_the_core(architecture):
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    model = prepare_model(reference, architecture=architecture(core=ToyCore)).build_model()
    assert model.core.reference is reference


def test_binding_a_reference_leaves_the_configuration_as_it_is(architecture):
    """The alleles are an input, not configuration: binding a reference gives the core its bank only."""
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    model = AlignAIR(config(architecture=architecture(core=ToyCore)))
    before = model.cfg
    model.bind_reference(reference)
    assert model.core.reference is reference and model.cfg is before
