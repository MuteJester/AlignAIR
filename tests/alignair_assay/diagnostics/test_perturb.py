"""``diagnostics.perturb``: a model file with one module's weights zeroed and every other byte unchanged, the positive
control of the readiness audit."""
from __future__ import annotations

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair.model_file import container, load_model, save_model, serialize  # noqa: E402

from alignair_assay.diagnostics.perturb import zero_module  # noqa: E402

pytestmark = pytest.mark.component

REFERENCE = "HUMAN_IGH_OGRDB"


@pytest.fixture
def checkpoint(tmp_path, make_small_model):
    model, reference_set = make_small_model(REFERENCE, seed=2)
    path = tmp_path / "base.alignair"
    save_model(str(path), model, reference_set, dataconfigs=[REFERENCE], training={"steps": 0, "batch_size": 0},
               include_trusted_pickle=False)
    return path


def test_zeroing_a_module_changes_its_weights_and_nothing_else(tmp_path, checkpoint):
    out = tmp_path / "perturbed.alignair"
    found = zero_module(checkpoint, out, "orientation_head.fc")
    decoded = serialize.state_dict_from_bytes(container.read_section(str(checkpoint), 'weights'))
    expected = [name for name in decoded if name.startswith('orientation_head.fc.')]
    assert set(expected) == {'orientation_head.fc.weight', 'orientation_head.fc.bias'}
    # sorted by name: safetensors hands the weights back in no fixed order (it varies from one load to the next)
    assert found["parameters"] == ["orientation_head.fc.bias", "orientation_head.fc.weight"]
    assert found["values"] > 0
    base, perturbed = load_model(str(checkpoint)).model, load_model(str(out)).model
    before, after = base.state_dict(), perturbed.state_dict()
    for name in before:
        if name.startswith("orientation_head.fc."):
            assert torch.count_nonzero(after[name]) == 0 and torch.count_nonzero(before[name]) > 0
        else:
            assert torch.equal(before[name], after[name]), name
    # every other section and the header are byte for byte the file's
    head_a, head_b = container.read_header(str(checkpoint)), container.read_header(str(out))
    for name in head_a["sections"]:
        if name != "weights":
            assert container.read_section(str(checkpoint), name) == container.read_section(str(out), name), name
    strip = ("sections", "_sections_base")
    assert {k: v for k, v in head_a.items() if k not in strip} == {k: v for k, v in head_b.items() if k not in strip}


def test_the_perturbed_file_is_deterministic(tmp_path, checkpoint):
    zero_module(checkpoint, tmp_path / "one.alignair", "orientation_head.fc")
    zero_module(checkpoint, tmp_path / "two.alignair", "orientation_head.fc")
    assert (tmp_path / "one.alignair").read_bytes() == (tmp_path / "two.alignair").read_bytes()
    assert (tmp_path / "one.alignair").read_bytes() != checkpoint.read_bytes()


def test_a_module_without_weights_is_refused(tmp_path, checkpoint):
    with pytest.raises(ValueError, match="no parameter of module 'core.missing_unit'"):
        zero_module(checkpoint, tmp_path / "out.alignair", "core.missing_unit")
    assert not (tmp_path / "out.alignair").exists()
