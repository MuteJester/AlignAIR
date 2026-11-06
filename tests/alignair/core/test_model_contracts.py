"""The model's construction contracts: architecture registration refuses what it cannot hold, a component factory
must give a torch module, the model takes only an AlignAIRConfig, an empty read is refused when the configuration
says so, and a model without an orientation head or bank vote runs its core on the read as given and counts the alleles of
the reference it is bound to."""
import pytest
import torch
from torch import nn

from alignair.core import AlignAIR, AlignAIRConfig, ModelComponents, build_model_config, register_architecture
from alignair.core import components
from alignair.errors import ConfigurationError, InputError

pytestmark = pytest.mark.component


class EchoCore(nn.Module):
    """Returns the tokens it is given, so a test sees what the model hands its core."""

    def __init__(self, cfg):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, emb, mask, end_mask, *, tokens, batch):
        return {"tokens": tokens, "emb": emb * self.scale}


def _objective(cfg):
    return None


@pytest.fixture
def scratch(monkeypatch):
    """Registrations made by the test are forgotten after it."""
    monkeypatch.setattr(components, "_ARCHITECTURES", dict(components._ARCHITECTURES))


def _config(**options):
    return build_model_config(3, 2, 2, has_d=True, max_seq_length=32, embed_dim=4, **options)


@pytest.mark.parametrize("name", ["", "  ", " test-padded-v1", None, 7])
def test_an_architecture_name_must_be_a_bare_nonempty_string(scratch, name):
    with pytest.raises(ConfigurationError, match="non-empty string"):
        register_architecture(name, ModelComponents(core=EchoCore, objective=_objective))


def test_registration_needs_model_components_and_never_overwrites(scratch):
    with pytest.raises(ConfigurationError, match="must be ModelComponents"):
        register_architecture("test-not-components-v1", {"core": EchoCore})
    register_architecture("test-echo-v1", ModelComponents(core=EchoCore, objective=_objective))
    with pytest.raises(ConfigurationError, match="already registered"):
        register_architecture("test-echo-v1", ModelComponents(core=EchoCore, objective=_objective))
    with pytest.raises(ConfigurationError, match="already registered"):
        register_architecture("chip-v1", components.CHIP)


def test_a_factory_that_gives_no_torch_module_is_refused_at_construction(scratch):
    register_architecture("test-no-module-v1", ModelComponents(core=lambda cfg: object(), objective=_objective))
    with pytest.raises(ConfigurationError, match="must return a torch module") as caught:
        AlignAIR(_config(architecture="test-no-module-v1", bank_orientation=False))
    assert caught.value.context["component"] == "core" and caught.value.context["returned_type"] == "object"


def test_an_unimportable_architecture_module_is_named(monkeypatch):
    monkeypatch.setenv(components.ARCHITECTURE_MODULES, "no_such_module_for_alignair")
    with pytest.raises(ConfigurationError, match="cannot be imported") as caught:
        components.architecture_components("test-unregistered-v1")
    assert caught.value.context["module"] == "no_such_module_for_alignair"


@pytest.mark.parametrize("config", [None, {"v_allele_count": 3}, "chip-v1"])
def test_the_model_takes_only_an_alignair_config(config):
    with pytest.raises(ConfigurationError, match="requires an AlignAIRConfig"):
        AlignAIR(config)


def test_an_empty_read_is_refused_when_the_configuration_says_so(scratch):
    register_architecture("test-echo-v1", ModelComponents(core=EchoCore, objective=_objective))
    tokens = torch.zeros(2, 32, dtype=torch.long)
    tokens[0, :5] = 1                                          # the second read is all padding
    strict = AlignAIR(_config(architecture="test-echo-v1", bank_orientation=False, reject_empty_sequences=True))
    with pytest.raises(InputError, match="empty"):
        strict({"tokenized_sequence": tokens})
    lenient = AlignAIR(_config(architecture="test-echo-v1", bank_orientation=False))
    assert lenient({"tokenized_sequence": tokens})["tokens"].shape == (2, 32)


def test_without_an_orientation_head_the_core_reads_the_tokens_as_given(scratch):
    register_architecture("test-echo-v1", ModelComponents(core=EchoCore, objective=_objective))
    model = AlignAIR(_config(architecture="test-echo-v1", use_orientation=False, bank_orientation=False)).eval()
    assert model.orientation_head is None and model.bank_orientation is None
    assert set(model.component_modules()) == {"embedding", "core"}
    tokens = torch.randint(1, 5, (2, 32))
    out = model({"tokenized_sequence": tokens})
    assert torch.equal(out["tokens"], tokens) and "orientation_logits" not in out and "orientation" not in out
    assert out["start_mask"].all() and not out["end_mask"][:, 0].any()
    from alignair.reference import ReferenceSet
    model.bind_reference(ReferenceSet.from_genotype({"V": {"V*01": "ACGT"}, "D": {"D*01": "GG"}, "J": {"J*01": "TT"}}))
    assert (model.cfg.v_allele_count, model.cfg.j_allele_count, model.cfg.d_allele_count) == (1, 1, 1)


def test_a_model_built_from_dataconfigs_takes_their_genes(scratch):
    import GenAIRR.data as gd
    register_architecture("test-echo-v1", ModelComponents(core=EchoCore, objective=_objective))
    model = AlignAIR.from_dataconfigs(gd.HUMAN_IGK_OGRDB, architecture="test-echo-v1", bank_orientation=False)
    assert not model.cfg.has_d and model.cfg.genes == ("v", "j")
    assert AlignAIR.from_config(model.cfg).cfg == model.cfg


def test_the_configurations_own_checks_name_their_field():
    for options, field in (({"trunk_channels": 0}, "trunk_channels"), ({"anchor_heads": 1}, "anchor_heads"),
                           ({"has_d": False, "d_presence_head": True}, "d_presence_head"),
                           ({"use_orientation": False}, "bank_orientation"), ({"input_space": "protein"}, "input_space")):
        with pytest.raises(ConfigurationError) as caught:
            AlignAIRConfig(2, 2, 2, **options)
        assert caught.value.context["field"] == field, options
