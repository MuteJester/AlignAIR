"""lab-trunk-v1: one shared, fully convolutional, length-preserving trunk scores every read position once, and each
gene's start and end logits are a 1x1 convolution of it. The classifiers and the meta tower are conv-v1's, on the
embedding, so the candidate changes segmentation only. It keeps conv-v1's outputs, learns, round-trips through a model
file, replays in the assay as its own forward pass and is screened against a baseline screen."""
import itertools
import json

import pytest
import torch

import GenAIRR.data as gd

from alignair import api
from alignair.core import prepare_model
from alignair.core.components import architecture_components
from alignair.reference.reference_set import ReferenceSet
from alignair.tasks.functional import soft_cross_entropy_rows, soft_gaussian_target
from alignair_lab import architectures  # noqa: F401  (registers lab-trunk-v1)
from alignair_lab.architectures.trunk import TRUNK_CHANNELS, DilatedTrunk, TrunkGeneBranch, build_trunk_heads
from tests.alignair_lab.rig import SMALL, candidate_arguments

GENES = ('v', 'd', 'j')
LENGTH = 576


@pytest.fixture(scope='module')
def reference():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def build(reference, architecture, seed=0, **options):
    torch.manual_seed(seed)
    return prepare_model(reference, architecture=architecture, **options).build_model().eval()


def tokens(lengths=(LENGTH, 380, 250), seed=1):
    generator = torch.Generator().manual_seed(seed)
    batch = torch.randint(1, 6, (len(lengths), LENGTH), generator=generator)
    for row, valid in enumerate(lengths):
        batch[row, valid:] = 0
    return {'tokenized_sequence': batch}


def records(n, seed=0):
    """``n`` GenAIRR IGH records with a V, a D and a J in the read, the D not inverted."""
    from alignair.train.gym import Curriculum, build_experiment
    parameters = {**Curriculum().params(0.3), 'invert_d_prob': 0.0}
    experiment = build_experiment(gd.HUMAN_IGH_OGRDB, parameters, allow_curatable=True)
    found = [record for record in itertools.islice(experiment.stream_records(n=None, seed=seed), 60)
             if all(record.get(f'{g}_sequence_start') is not None for g in GENES)][:n]
    assert len(found) == n
    return found


def count(module):
    return sum(parameter.numel() for parameter in module.parameters())


# --- the trunk --------------------------------------------------------------------------------------------------

@pytest.mark.component
def test_the_trunk_keeps_length_and_zeroes_padding():
    torch.manual_seed(0)
    trunk = DilatedTrunk(8)
    emb = torch.randn(3, LENGTH, 8)
    mask = torch.arange(LENGTH) < torch.tensor([LENGTH, 380, 250]).unsqueeze(1)
    noisy = torch.where(mask.unsqueeze(-1), emb, torch.randn_like(emb))     # other values on the pad
    with torch.no_grad():
        trained = trunk.train()(emb, mask)
        out, other = trunk.eval()(emb, mask), trunk(noisy, mask)
    assert out.shape == (3, LENGTH, TRUNK_CHANNELS)
    assert not out[~mask].any() and not trained[~mask].any()
    assert out[mask].abs().sum(-1).gt(0).all()
    assert torch.equal(out, other)                                          # the pad never reaches the read


@pytest.mark.component
def test_the_trunk_sees_505_positions():
    """One input position moved changes the outputs of 505 positions around it (1 + 4 x 126), clipped to the read."""
    torch.manual_seed(0)
    trunk = DilatedTrunk(8).eval()
    emb = torch.randn(1, LENGTH, 8)
    for valid, probe, seen in ((LENGTH, 288, (288 - 252, 288 + 252)), (300, 280, (280 - 252, 299))):
        mask = (torch.arange(LENGTH) < valid).unsqueeze(0)
        moved = emb.clone()
        moved[0, probe] += 3.0
        with torch.no_grad():
            changed = (trunk(moved, mask) != trunk(emb, mask)).any(-1)[0].nonzero().flatten().tolist()
        assert changed == list(range(seen[0], seen[1] + 1)), (valid, probe)
    assert len(range(288 - 252, 288 + 253)) == 505


# --- lab-trunk-v1 -----------------------------------------------------------------------------------------------

@pytest.mark.component
def test_lab_trunk_v1_has_conv_v1_output_keys(reference):
    conv, trunk = build(reference, 'conv-v1', **SMALL), build(reference, 'lab-trunk-v1', **SMALL)
    with torch.no_grad():
        expected, out = conv(tokens()), trunk(tokens())
    assert {key: tuple(value.shape) for key, value in out.items()} == \
           {key: tuple(value.shape) for key, value in expected.items()}
    assert build_trunk_heads(trunk.cfg) == {} and len(trunk.trunk_heads) == 0
    points = trunk.component_modules()
    assert points['trunk'] is trunk.trunk
    for g in GENES:
        assert isinstance(trunk.branches[g], TrunkGeneBranch)
        assert {name for name in points if name.startswith(f'{g}.')} == \
               {f'{g}.{name}' for name in ('boundary', 'cutout', 'cls_tower', 'cls_mid', 'cls_head')}


@pytest.mark.component
def test_lab_trunk_v1_classifies_and_reads_metadata_as_conv_v1(reference):
    """Given conv-v1's weights, its classifiers and its meta tower give conv-v1's outputs from the embedding."""
    conv, trunk = build(reference, 'conv-v1', **SMALL), build(reference, 'lab-trunk-v1', seed=1, **SMALL)
    emb = conv.embedding(tokens()['tokenized_sequence'])
    features = torch.randn(3, LENGTH, TRUNK_CHANNELS)
    starts, ends = torch.tensor([[3.5], [40.0], [100.2]]), torch.tensor([[290.0], [200.7], [101.0]])
    with torch.no_grad():
        for g in GENES:
            ours, theirs = trunk.branches[g], conv.branches[g]
            loaded = ours.load_state_dict(theirs.state_dict(), strict=False)
            assert sorted(loaded.missing_keys) == ['boundary.bias', 'boundary.weight']
            assert {key.split('.')[0] for key in loaded.unexpected_keys} == {'seg_tower', 'start_head', 'end_head'}
            assert torch.equal(ours.classify(emb, starts, ends, features=features), theirs.classify(emb, starts, ends))
        trunk.meta_tower.tower.load_state_dict(conv.meta_tower.state_dict())
        assert torch.equal(trunk.meta_tower(emb, features=features), conv.meta_tower(emb))
    assert type(trunk.meta_tower.tower) is type(architecture_components('conv-v1').meta_tower(conv.cfg))


@pytest.mark.component
def test_lab_trunk_v1_segmentation_is_a_fraction_of_conv_v1s(reference):
    """The trunk and its 3 boundary convolutions hold under a quarter of the parameters of conv-v1's 3 segmentation
    towers and 6 dense boundary heads."""
    ours = build(reference, 'lab-trunk-v1').component_modules()
    theirs = build(reference, 'conv-v1').component_modules()
    trunk = count(ours['trunk']) + sum(count(ours[f'{g}.boundary']) for g in GENES)
    segmentation = sum(count(theirs[f'{g}.{name}']) for g in GENES for name in ('seg_tower', 'start_head', 'end_head'))
    assert trunk < segmentation / 4


@pytest.mark.integration
@pytest.mark.slow
def test_lab_trunk_v1_learns_one_batch(reference):
    """Updated 200 times on one batch of 4 GenAIRR reads, its boundary loss (the soft-Gaussian cross entropy of the
    6 boundaries, in eval mode) falls by at least half."""
    from alignair.train import overfit_one_batch
    from alignair.train.trainer import build_batch
    model = build(reference, 'lab-trunk-v1', **SMALL)
    batch, targets = build_batch(records(4), reference, model.cfg)

    def boundary_loss():
        with torch.no_grad():
            out = model.eval()(batch)
        losses = [soft_cross_entropy_rows(out[f'{key}_logits'], soft_gaussian_target(targets[key], LENGTH),
                                          out['position_mask']).mean()
                  for key in (f'{g}_{edge}' for g in GENES for edge in ('start', 'end'))]
        return float(sum(losses) / len(losses))

    before = boundary_loss()
    overfit_one_batch(model, batch, targets, steps=200, lr=1e-3)
    assert boundary_loss() <= before / 2


# --- the model file, the replay and the screen -------------------------------------------------------------------

@pytest.mark.integration
def test_a_lab_trunk_v1_model_file_round_trips(reference, tmp_path, monkeypatch):
    """Written as a training run writes its model.alignair, it loads back (in a process that registered the lab's
    candidates) and predicts what it predicted; a process that did not is refused, naming the architecture."""
    from alignair.core import components
    from alignair.errors import ConfigurationError
    from alignair.train.checkpoints import save_checkpoint
    model = build(reference, 'lab-trunk-v1', **SMALL)
    path = tmp_path / 'model.alignair'
    save_checkpoint(path, model.cfg, model, None, 7, dataconfigs=[gd.HUMAN_IGH_OGRDB])
    loaded, loaded_reference = api.load_model(str(path))
    assert loaded.cfg == model.cfg and loaded.cfg.architecture == 'lab-trunk-v1'
    assert list(loaded.state_dict()) == list(model.state_dict())
    with torch.no_grad():
        first, second = model(tokens()), loaded(tokens())
    assert all(torch.equal(first[key], second[key]) for key in first)
    sequences = [record['sequence'] for record in records(3, seed=11)]
    assert api.predict_sequences(loaded, loaded_reference, sequences, device='cpu') == \
           api.predict_sequences(model, reference, sequences, device='cpu')
    monkeypatch.setattr(components, '_ARCHITECTURES',
                        {name: value for name, value in components._ARCHITECTURES.items() if name != 'lab-trunk-v1'})
    with pytest.raises(ConfigurationError, match='lab-trunk-v1'):
        api.load_model(str(path))


@pytest.mark.integration
def test_the_replay_of_lab_trunk_v1_is_its_forward_pass(reference, monkeypatch):
    """The assay's replay gives what the product predicts; fed the model's own spans, its classifier call recomputes
    the trunk's features from the re-embedded read, as the forward pass computes them, and gives the forward pass's
    allele probabilities."""
    from alignair_assay.diagnostics.replay import Substitutions, replay
    model = build(reference, 'lab-trunk-v1', **SMALL)
    sequences = [record['sequence'] for record in records(5, seed=11)]
    ids = [f'read{k}' for k in range(len(sequences))]
    base = replay(model, reference, sequences, ids, device='cpu', batch_size=2)
    assert base.airr_rows == api.predict_sequences(model, reference, sequences, device='cpu', batch_size=2)

    trunk_calls, classified = [], []
    model.trunk.register_forward_hook(lambda module, inputs, output: trunk_calls.append((inputs, output)))
    for g in GENES:
        original = model.branches[g].classify

        def spy(emb, starts, ends, *, features, original=original):
            classified.append(features)
            return original(emb, starts, ends, features=features)
        monkeypatch.setattr(model.branches[g], 'classify', spy)
    none = [None] * len(sequences)
    out = replay(model, reference, sequences, ids, device='cpu', batch_size=2,
                 substitutions=Substitutions(spans={g: (none, none) for g in GENES}, classifier_spans=True))
    assert out.airr_rows == base.airr_rows
    assert set(out.raw) == set(base.raw) and all((out.raw[key] == base.raw[key]).all() for key in base.raw)
    batches = -(-len(sequences) // 2)
    assert len(trunk_calls) == 2 * batches and len(classified) == 2 * len(GENES) * batches
    for k in range(batches):
        (forward_inputs, forward_features), (replay_inputs, replay_features) = trunk_calls[2 * k:2 * k + 2]
        assert all(torch.equal(a, b) for a, b in zip(forward_inputs, replay_inputs))
        assert torch.equal(forward_features, replay_features)
        calls = classified[2 * len(GENES) * k:2 * len(GENES) * (k + 1)]
        assert all(features is forward_features for features in calls[:len(GENES)])
        assert all(features is replay_features for features in calls[len(GENES):])


@pytest.mark.integration
def test_a_screen_of_lab_trunk_v1_compares_with_the_baseline_screen(rig, baseline):
    """The screen trains the candidate, then the assay diagnoses its model file in the screen's process."""
    from alignair.model_file import read_model_state
    from alignair_lab.cli import main
    assert main(candidate_arguments(rig, 'trunk', architecture='lab-trunk-v1')) == 0
    record = json.loads((rig.screens / 'trunk' / 'screen.json').read_text())
    assert (record['status'], record['architecture'], record['baseline']['name']) == ('done', 'lab-trunk-v1', 'base0')
    assert read_model_state(record['candidate']['model_path']).config.architecture == 'lab-trunk-v1'
    assert set(record['comparison']['suites']) == {'S0'} and record['candidate']['model_key']
