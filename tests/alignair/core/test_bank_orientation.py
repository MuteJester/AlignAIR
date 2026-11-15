"""Bank orientation: at inference, each read's orientation is the transform under which its 12-mers are most found in
the bound reference's germlines; the orientation head decides only when no transform wins."""
import pytest
import torch

import GenAIRR.data as gd

from alignair.core import AlignAIR, AlignAIRConfig
from alignair.errors import ConfigurationError
from alignair.network.orientation import NUM_ORIENTATIONS, apply_orientation
from alignair.reference.reference_set import ReferenceSet
from alignair.core.tokens import TOKEN_DICT

pytestmark = pytest.mark.component

LENGTH = 128


@pytest.fixture(scope="module")
def reference():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def _model(reference, *, bind=True, **options):
    torch.manual_seed(0)
    cfg = AlignAIRConfig(has_d=True, max_seq_length=LENGTH, embed_dim=4, trunk_channels=8, **options)
    model = AlignAIR(cfg).eval()
    if bind:
        model.bind_reference(reference)
    return model


def _head_says(model, orientation):
    """The orientation head calls ``orientation`` for every read, whatever it reads."""
    with torch.no_grad():
        model.orientation_head.fc.weight.zero_()
        model.orientation_head.fc.bias.fill_(-10.)
        model.orientation_head.fc.bias[orientation] = 10.


def _tokens(sequences):
    tokens = torch.zeros(len(sequences), LENGTH, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor([TOKEN_DICT[base] for base in sequence])
    return tokens


def _re_embedded(model, batch):
    """The tokens the model re-embeds after its orientation correction (the embedding's second call)."""
    seen = []
    handle = model.embedding.register_forward_hook(lambda module, args, output: seen.append(args[0].clone()))
    try:
        out = model(batch)
    finally:
        handle.remove()
    return out, seen[1]


def test_the_bank_vote_corrects_a_read_the_head_misorients(reference):
    model = _model(reference, bank_orientation=True)
    _head_says(model, 0)
    forward = _tokens([reference.gene("V").sequences[0][-110:]] * NUM_ORIENTATIONS)
    mask = forward != 0
    presented = apply_orientation(forward, mask, torch.arange(NUM_ORIENTATIONS))
    out, corrected = _re_embedded(model, {"tokenized_sequence": presented})
    assert out["orientation"].tolist() == list(range(NUM_ORIENTATIONS))
    assert torch.equal(corrected, forward)


def test_a_read_without_bank_kmers_keeps_the_heads_call(reference):
    model = _model(reference, bank_orientation=True)
    _head_says(model, 2)
    out = model({"tokenized_sequence": _tokens(["ACGTAN" * 18])})      # every 12-mer holds an N: no hit anywhere
    assert out["orientation"].tolist() == [2]


def test_a_teacher_forced_orientation_is_applied_as_given(reference):
    model = _model(reference, bank_orientation=True)
    forward = _tokens([reference.gene("V").sequences[0][-110:]])
    _, corrected = _re_embedded(model, {"tokenized_sequence": forward, "orientation": torch.tensor([1])})
    assert torch.equal(corrected, apply_orientation(forward, forward != 0, torch.tensor([1])))


def test_without_the_option_the_head_decides(reference):
    model = _model(reference, bank_orientation=False)
    _head_says(model, 0)
    forward = _tokens([reference.gene("V").sequences[0][-110:]])
    presented = apply_orientation(forward, forward != 0, torch.tensor([1]))
    assert model({"tokenized_sequence": presented})["orientation"].tolist() == [0]


def test_the_vote_needs_a_bound_reference(reference):
    model = _model(reference, bind=False, bank_orientation=True)
    with pytest.raises(ConfigurationError, match="bind_reference"):
        model({"tokenized_sequence": _tokens(["ACGT" * 20])})


def test_the_bank_index_is_not_saved_with_the_weights(reference):
    model = _model(reference, bank_orientation=True)
    assert not [key for key in model.state_dict() if key.startswith("bank_orientation")]


def test_the_option_is_a_boolean():
    with pytest.raises(ConfigurationError) as caught:
        AlignAIRConfig(has_d=True, max_seq_length=LENGTH, bank_orientation=1)
    assert caught.value.context["field"] == "bank_orientation"
