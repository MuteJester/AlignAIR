"""The network's outputs are read once, into typed model outputs (``predict.outputs.ModelOutputs``): every key a chip,
a light-chain model and a model without optional heads emit is read and given back unchanged; a missing required
head, a key the pipeline does not know and half of a pair are refused; a light chain keeps its parser's D logits."""
import json

import GenAIRR.data as gd
import pytest
import torch

from alignair.core.anchors import ANCHOR_LOGITS
from alignair.core.d_presence import D_PRESENCE_LOGITS
from alignair.errors import InferenceError
from alignair.predict import PredictConfig
from alignair.predict.forward import run_model
from alignair.predict.outputs import ModelOutputs
from alignair.reference.reference_set import ReferenceSet
from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP

pytestmark = pytest.mark.component

READS = json.loads(FIXTURE.read_text())["reads"][:4]


def _chip():
    from alignair.api import load_model
    return load_model(str(TEST_CHIP), device="cpu")


def _small(reference, **options):
    from alignair.core import prepare_model
    torch.manual_seed(0)
    return prepare_model(reference, embed_dim=4, trunk_channels=8, **options).build_model().eval(), reference


MODELS = {
    "chip": _chip,
    "igk": lambda: _small(ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB)),
}


class _WithoutOptionalHeads(torch.nn.Module):
    """A core registered through ``core.components`` may give no anchor or D presence logits: the chip with them
    dropped."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.cfg = inner, inner.cfg

    def forward(self, batch):
        out = self.inner(batch)
        for key in (*ANCHOR_LOGITS.values(), D_PRESENCE_LOGITS):
            out.pop(key, None)
        return out


def _batch(model_and_reference):
    model, reference = model_and_reference
    cfg = PredictConfig.for_model(model, reference, batch_size=8)
    return run_model(model, READS, cfg)[0], cfg.genes


@pytest.fixture(scope="module", params=list(MODELS))
def model(request):
    return MODELS[request.param]()


@pytest.fixture(scope="module")
def chip_batch():
    return _batch(MODELS["chip"]())[0]


def test_a_models_outputs_are_read_in_full(model):
    batch, genes = _batch(model)
    back = ModelOutputs.from_network(batch, genes).network_dict()
    assert back.keys() == batch.keys() and all(back[key] is batch[key] for key in batch)


def test_outputs_without_the_optional_ones_are_given_back_without_them(chip_batch):
    """A core registered through ``core.components`` may give no allele log posteriors, anchor logits or D presence
    logits: the outputs read without them give back exactly the keys they were read from."""
    optional = {*ANCHOR_LOGITS.values(), D_PRESENCE_LOGITS, *(f"{gene}_log_posterior" for gene in ("v", "d", "j"))}
    batch = {key: value for key, value in chip_batch.items() if key not in optional}
    back = ModelOutputs.from_network(batch, ("v", "d", "j")).network_dict()
    assert back.keys() == batch.keys() and all(back[key] is batch[key] for key in batch)


def test_a_missing_head_output_is_refused(chip_batch):
    for key in ("v_allele", "indel_count"):
        with pytest.raises(InferenceError, match=key):
            ModelOutputs.from_network({k: v for k, v in chip_batch.items() if k != key}, ("v", "d", "j"))


def test_an_output_the_pipeline_does_not_know_is_refused(chip_batch):
    with pytest.raises(InferenceError, match="chip_log_z"):
        ModelOutputs.from_network({**chip_batch, "chip_log_z": chip_batch["indel_count"]}, ("v", "d", "j"))


@pytest.mark.parametrize("dropped, named", [(("v_start_logits",), "v_end_logits"),
                                            (("v_anchor_logits",), "j_anchor_logits"),
                                            (("start_mask", "end_mask"), "start_mask")])
def test_half_pairs_are_refused(chip_batch, dropped, named):
    with pytest.raises(InferenceError, match=named):
        ModelOutputs.from_network({k: v for k, v in chip_batch.items() if k not in dropped}, ("v", "d", "j"))


def test_a_light_chain_keeps_its_parsers_d_logits():
    batch, genes = _batch(MODELS["igk"]())
    out = ModelOutputs.from_network(batch, genes)
    assert genes == ("v", "j") and "d" not in out.genes and "d" in out.boundary_logits


def test_a_core_without_the_optional_heads_predicts_rows_without_them():
    from alignair.api import predict_sequences
    model, reference = _small(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB))
    rows = predict_sequences(_WithoutOptionalHeads(model), reference, READS, device="cpu")
    assert rows and all(row["productive"] is None for row in rows)
    assert not any("v_anchor" in row or "d_presence" in row for row in rows)
