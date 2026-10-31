"""lab-trunk-v1: one shared, fully convolutional, length-preserving trunk scores every read position once, and each
gene's start and end logits are a 1x1 convolution of it. The classifiers and the meta tower are conv-v1's, on the
embedding, so the candidate changes segmentation only. It keeps conv-v1's outputs, learns, round-trips through a model
file, replays in the assay as its own forward pass and is screened against a baseline screen.

lab-trunk-v2 is lab-trunk-v1 for half-open boundaries: its end slot k reads the features of base k - 1, so a segment
ending at the read's last base is supervised at slot n from that base's features.

With the model option region_head=true, either also labels every read position with its recombination region from the
trunk's features, and learns to."""
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
from alignair_lab.architectures.trunk import (
    CROP_MARGIN,
    CROP_WINDOWS,
    TRUNK_CHANNELS,
    DilatedTrunk,
    TrunkGeneBranch,
    build_trunk_heads,
    crop_window,
)
from tests.alignair_lab.rig import SMALL, candidate_arguments, flat

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
def test_a_reads_features_do_not_depend_on_the_padding_around_it():
    """With LayerNorm's bias non-zero, a read of n > L - 64 positions (fewer pads after it than a dilation-32 tap
    reaches across) has the same features whether it fills its window or is followed by L - n or L + 64 - n pads,
    or has pads on both sides: a convolution reads a pad position as it reads the zero padding past the window."""
    torch.manual_seed(0)
    trunk = DilatedTrunk(8).eval()
    with torch.no_grad():
        for block in trunk.blocks:
            block.norm.bias.fill_(1.0)
    n = LENGTH - 36
    read = torch.randn(1, n, 8)

    def features(width, before=0):
        emb = torch.randn(1, width, 8)                                      # other values on the pads
        emb[:, before:before + n] = read
        mask = torch.zeros(1, width, dtype=torch.bool)
        mask[:, before:before + n] = True
        with torch.no_grad():
            return trunk(emb, mask)[0, before:before + n]

    alone = features(n)
    for width, before in ((LENGTH, 0), (LENGTH + 64, 0), (LENGTH + 64, 32)):
        assert torch.allclose(features(width, before), alone, rtol=1e-5, atol=1e-5), (width, before)


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


# --- lab-trunk-v2: half-open boundaries ---------------------------------------------------------------------------

def half_open(reference, seed=0):
    return build(reference, 'lab-trunk-v2', seed=seed, boundary_mode='half_open', **SMALL)


@pytest.mark.component
def test_lab_trunk_v2_reads_each_end_slot_from_the_segments_last_base(reference):
    """A start logit at slot k is the head on the features of base k, as in lab-trunk-v1. An end logit at slot k (the
    half-open end of a segment whose last base is k - 1) is the head on the features of base k - 1, where lab-trunk-v1
    reads base k. Slot 0, never an end, holds one finite constant."""
    from alignair.core.boundaries import boundary_masks
    ours = half_open(reference).branches['j']
    v1 = build(reference, 'lab-trunk-v1', **SMALL).branches['j']
    v1.load_state_dict(ours.state_dict())
    start_mask, end_mask = boundary_masks(tokens((LENGTH - 1, 380, 250))['tokenized_sequence'], 'half_open')
    features = torch.randn(3, LENGTH, TRUNK_CHANNELS)
    with torch.no_grad():
        heads = features @ ours.boundary.weight[:, :, 0].T + ours.boundary.bias      # (B, L, 2): both heads per base
        s_log, e_log, s_exp, e_exp = ours.segment(None, start_mask, end_mask, features=features)
        residue = v1.segment(None, start_mask, start_mask, features=features)[1]
    assert torch.allclose(s_log, heads[..., 0], atol=1e-5)
    assert torch.allclose(e_log[:, 1:], heads[:, :-1, 1], atol=1e-5)
    assert torch.allclose(residue, heads[..., 1], atol=1e-5)
    assert torch.isfinite(e_log).all() and (e_log[:, 0] == e_log[0, 0]).all()
    assert torch.isfinite(s_exp).all() and torch.isfinite(e_exp).all()


@pytest.mark.component
def test_lab_trunk_v2_refuses_residue_boundaries(reference):
    from alignair.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match='residue') as caught:
        build(reference, 'lab-trunk-v2', **SMALL)
    assert caught.value.context['field'] == 'boundary_mode'


@pytest.mark.integration
def test_lab_trunk_v2_supervises_an_end_at_the_reads_last_base(reference):
    """A GenAIRR read whose J ends at its last base has its J end target at slot n. The loss is finite, and the J end
    cross entropy's gradient reaches the trunk's features of base n - 1, which only slot n reads."""
    from alignair.tasks import build_tasks
    from alignair.tasks.losses import masked_soft_gaussian
    from alignair.train.trainer import build_batch
    record = next(record for record in records(8) if record['j_sequence_end'] == len(record['sequence']))
    n = len(record['sequence'])
    model = half_open(reference)
    batch, targets = build_batch([record], reference, model.cfg)
    assert targets['j_end'].item() == n
    trunk_outputs = []

    def keep_gradient(module, inputs, output):
        output.retain_grad()
        trunk_outputs.append(output)
    model.trunk.register_forward_hook(keep_gradient)
    out = model(batch)
    tasks = build_tasks(model.cfg)
    assert torch.isfinite(tasks.loss(out, targets, tasks.make_weights())[0])
    target = masked_soft_gaussian(targets['j_end'], LENGTH, out['end_mask'])
    soft_cross_entropy_rows(out['j_end_logits'], target, out['end_mask']).sum().backward()
    gradient = trunk_outputs[0].grad[0].abs().sum(-1)
    assert gradient[n - 1] > 0 and not gradient[n:].any()


@pytest.mark.integration
@pytest.mark.slow
def test_lab_trunk_v2_learns_the_j_ends_of_one_batch(reference):
    """Updated 200 times on one batch of 8 GenAIRR reads, its J end decoded as the product decodes it (the argmax of
    the allowed end slots) is exact on more than 90% of them."""
    from alignair.predict.decode import decoded_boundaries
    from alignair.train import overfit_one_batch
    from alignair.train.trainer import build_batch
    model = half_open(reference)
    batch, targets = build_batch(records(8), reference, model.cfg)
    overfit_one_batch(model, batch, targets, steps=200, lr=1e-3)
    with torch.no_grad():
        out = model.eval()(batch)
    _, ends = decoded_boundaries(out, 'j')
    assert (ends == targets['j_end'].reshape(-1).round().long()).float().mean() > 0.9


# --- the region head (region_head=true) --------------------------------------------------------------------------

@pytest.mark.component
@pytest.mark.parametrize('architecture,options', [('lab-trunk-v1', {}),
                                                  ('lab-trunk-v2', {'boundary_mode': 'half_open'})])
def test_the_region_head_is_one_1x1_convolution_of_the_trunks_features(reference, architecture, options):
    """With region_head on, a trunk architecture gives region logits (B, L, 7) over pre, V, N1, D, N2, J and post: one
    1x1 convolution (903 parameters) of the trunk's features. Built after every other module, it leaves the rest of the
    model, its initial weights and its outputs, as they are with the flag off."""
    from alignair.neural.heads.region import READ_REGIONS
    off = build(reference, architecture, **SMALL, **options)
    on = build(reference, architecture, **SMALL, **options, region_head=True)
    head = on.trunk_heads['region_logits']
    assert set(build_trunk_heads(on.cfg)) == {'region_logits'} and build_trunk_heads(off.cfg) == {}
    assert count(head) == TRUNK_CHANNELS * len(READ_REGIONS) + len(READ_REGIONS) == 903
    assert on.component_modules()['trunk_heads.region_logits'] is head
    assert set(on.state_dict()) - set(off.state_dict()) == {f'trunk_heads.region_logits.{key}'
                                                            for key in head.state_dict()}
    assert all(torch.equal(value, on.state_dict()[key]) for key, value in off.state_dict().items())
    features = []
    on.trunk.register_forward_hook(lambda module, inputs, output: features.append(output))
    with torch.no_grad():
        expected, out = off(tokens()), on(tokens())
    assert set(out) - set(expected) == {'region_logits'}
    assert all(torch.equal(out[key], expected[key]) for key in expected)
    assert out['region_logits'].shape == (3, LENGTH, len(READ_REGIONS))
    weight, bias = head.conv.weight[:, :, 0], head.conv.bias
    assert torch.allclose(out['region_logits'], features[0] @ weight.T + bias, atol=1e-5)


@pytest.mark.integration
@pytest.mark.slow
def test_lab_trunk_v1_with_regions_learns_one_batch(reference):
    """Updated 200 times on one batch of 4 GenAIRR reads, its region head (in eval mode) labels more than 90% of the
    reads' positions with their region."""
    from alignair.train import overfit_one_batch
    from alignair.train.trainer import build_batch
    model = build(reference, 'lab-trunk-v1', **SMALL, region_head=True)
    batch, targets = build_batch(records(4), reference, model.cfg)
    labels = targets['region_labels']

    def accuracy():
        with torch.no_grad():
            predicted = model.eval()(batch)['region_logits'].argmax(-1)
        read = labels != -100
        return float((predicted[read] == labels[read]).float().mean())

    before = accuracy()
    result = overfit_one_batch(model, batch, targets, steps=200, lr=1e-3)
    assert 'regions' in result['history'][0]
    assert accuracy() > 0.9 > before


# --- the anchor heads (anchor_heads=true) -------------------------------------------------------------------------

ANCHOR_KEYS = ('v_anchor_logits', 'j_anchor_logits')


@pytest.mark.component
@pytest.mark.parametrize('architecture,options', [('lab-trunk-v1', {}),
                                                  ('lab-trunk-v2', {'boundary_mode': 'half_open'})])
def test_the_anchor_heads_score_every_position_and_a_null_slot_from_the_trunks_features(reference, architecture,
                                                                                         options):
    """With anchor_heads on, a trunk architecture gives V and J anchor logits (B, L + 1): slot k < L is a 1x1
    convolution of position k's trunk features, slot L ("not observed") a linear map of the read's masked mean and max
    features; 386 parameters a head. The learned productive head is not built. Given the flag-off model's shared
    weights, every other output is the flag-off model's."""
    off = build(reference, architecture, **SMALL, **options)
    on = build(reference, architecture, **SMALL, **options, anchor_heads=True)
    assert set(build_trunk_heads(on.cfg)) == set(ANCHOR_KEYS) and build_trunk_heads(off.cfg) == {}
    assert 'productive' in off.meta_heads and 'productive' not in on.meta_heads
    for key in ANCHOR_KEYS:
        assert count(on.trunk_heads[key]) == (TRUNK_CHANNELS + 1) + (2 * TRUNK_CHANNELS + 1) == 386
        assert on.component_modules()[f'trunk_heads.{key}'] is on.trunk_heads[key]
    productive = {key for key in off.state_dict() if key.startswith('meta_heads.productive.')}
    anchors = {key for key in on.state_dict() if key.startswith('trunk_heads.')}
    assert productive and set(on.state_dict()) == set(off.state_dict()) - productive | anchors
    on.load_state_dict({**off.state_dict(), **{key: on.state_dict()[key] for key in anchors}}, strict=False)
    features = []
    on.trunk.register_forward_hook(lambda module, inputs, output: features.append(output))
    batch = tokens()
    with torch.no_grad():
        expected, out = off(batch), on(batch)
    assert set(expected) - set(out) == {'productive'} and set(out) - set(expected) == set(ANCHOR_KEYS)
    assert all(torch.equal(out[key], expected[key]) for key in out if key in expected)
    mask = batch['tokenized_sequence'] != 0
    feats, keep = features[0], mask.unsqueeze(-1)
    pooled = torch.cat([(feats * keep).sum(1) / keep.sum(1), feats.masked_fill(~keep, float('-inf')).amax(1)], -1)
    for key in ANCHOR_KEYS:
        head, logits = on.trunk_heads[key], out[key]
        assert logits.shape == (3, LENGTH + 1)
        position = feats @ head.position.weight[0, :, 0] + head.position.bias
        assert torch.allclose(logits[:, :LENGTH], position, atol=1e-5)
        assert torch.allclose(logits[:, LENGTH], head.null(pooled)[:, 0], atol=1e-5)


@pytest.mark.integration
@pytest.mark.slow
def test_lab_trunk_v2_with_anchors_learns_one_batch(reference):
    """Updated 200 times on one batch of 8 GenAIRR reads (6 full-length, 2 that miss their V Cys), its anchors
    decoded as the product decodes them (``decode_anchors``: the argmax over the codon starts and the null slot) are
    right for more than 90% of the 16 anchors, null ones included."""
    from alignair.predict.decode import decode_anchors
    from alignair.train import overfit_one_batch
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    from alignair.train.trainer import build_batch
    j_anchored = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)[4]
    lost = [record for record in itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, j_anchored, 5), 40)
            if record['junction_start'] is None][:2]
    model = build(reference, 'lab-trunk-v2', boundary_mode='half_open', **SMALL, anchor_heads=True)
    batch, targets = build_batch(records(6) + lost, reference, model.cfg)
    assert (targets['v_anchor'] == LENGTH).sum() >= 2 and (targets['v_anchor'] < LENGTH).sum() >= 5

    def accuracy():
        with torch.no_grad():
            out = model.eval()(batch)
        right = [(decode_anchors(out[f'{key}_logits'], out['start_mask']) == targets[key]).float()
                 for key in ('v_anchor', 'j_anchor')]
        return float(torch.cat(right).mean())

    before = accuracy()
    result = overfit_one_batch(model, batch, targets, steps=200, lr=1e-3)
    assert {'v_anchor', 'j_anchor'} <= set(result['history'][0]) and 'productive' not in result['history'][0]
    assert accuracy() > 0.9 > before


# --- the D presence head (d_presence_head=true) -------------------------------------------------------------------

@pytest.mark.component
@pytest.mark.parametrize('architecture,options', [('lab-trunk-v1', {}),
                                                  ('lab-trunk-v2', {'boundary_mode': 'half_open'})])
def test_the_d_presence_head_classifies_the_reads_pooled_trunk_features(reference, architecture, options):
    """With d_presence_head on, a trunk architecture gives D presence logits (B, 3) over absent, too short to call and
    callable: the read's masked mean and max trunk features through Linear(2C, 64), GELU and Linear(64, 3), 16,643
    parameters. Built after every other module, it leaves the rest of the model, its initial weights and its outputs,
    as they are with the flag off."""
    off = build(reference, architecture, **SMALL, **options)
    on = build(reference, architecture, **SMALL, **options, d_presence_head=True)
    head = on.trunk_heads['d_presence_logits']
    assert set(build_trunk_heads(on.cfg)) == {'d_presence_logits'}
    assert count(head) == (2 * TRUNK_CHANNELS * 64 + 64) + (64 * 3 + 3) == 16643
    assert on.component_modules()['trunk_heads.d_presence_logits'] is head
    assert set(on.state_dict()) - set(off.state_dict()) == {f'trunk_heads.d_presence_logits.{key}'
                                                            for key in head.state_dict()}
    assert all(torch.equal(value, on.state_dict()[key]) for key, value in off.state_dict().items())
    features = []
    on.trunk.register_forward_hook(lambda module, inputs, output: features.append(output))
    batch = tokens()
    with torch.no_grad():
        expected, out = off(batch), on(batch)
    assert set(out) - set(expected) == {'d_presence_logits'}
    assert all(torch.equal(out[key], expected[key]) for key in expected)
    keep = (batch['tokenized_sequence'] != 0).unsqueeze(-1)
    pooled = torch.cat([(features[0] * keep).sum(1) / keep.sum(1),
                        features[0].masked_fill(~keep, float('-inf')).amax(1)], -1)
    first, second = head.hidden, head.classes
    assert out['d_presence_logits'].shape == (3, 3)
    assert torch.allclose(out['d_presence_logits'], second(torch.nn.functional.gelu(first(pooled))), atol=1e-5)


@pytest.mark.integration
@pytest.mark.slow
def test_lab_trunk_v2_with_d_presence_learns_one_batch(reference):
    """Updated 200 times on one batch of 8 GenAIRR reads with every class (4 callable, 2 too short, 2 without a D), its
    D presence (the argmax of the head) is right for more than 90% of them."""
    from alignair.data.targets import read_d_presence
    from alignair.train import overfit_one_batch
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    from alignair.train.trainer import build_batch
    components = CurriculumStreamPolicy().parameters_for(gd.HUMAN_IGH_OGRDB)
    pool = [*itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[2], 41), 300),
            *itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, components[4], 42), 40)]
    by_class = {k: [r for r in pool if read_d_presence(r) == k] for k in range(3)}
    chosen = by_class[2][:4] + by_class[1][:2] + by_class[0][:2]
    assert len(chosen) == 8
    model = build(reference, 'lab-trunk-v2', boundary_mode='half_open', **SMALL, d_presence_head=True)
    batch, targets = build_batch(chosen, reference, model.cfg)
    assert targets['d_presence'].tolist() == [2] * 4 + [1] * 2 + [0] * 2

    def accuracy():
        with torch.no_grad():
            predicted = model.eval()(batch)['d_presence_logits'].argmax(-1)
        return float((predicted == targets['d_presence']).float().mean())

    before = accuracy()
    result = overfit_one_batch(model, batch, targets, steps=200, lr=1e-3)
    assert 'd_presence' in result['history'][0]
    assert accuracy() > 0.9 > before


# --- lab-trunk-crop-v1: each gene's classifier on a crop of the trunk's features ----------------------------------

def crop_model(reference, seed=0, **options):
    return build(reference, 'lab-trunk-crop-v1', seed=seed, boundary_mode='half_open', **SMALL, **options)


def positions(lengths=(LENGTH,), channels=3):
    """Features whose every channel holds the position's index + 1 (0 marks nothing gathered), and the read mask of
    reads of ``lengths``, each starting at position 0."""
    batch = len(lengths)
    features = (torch.arange(LENGTH, dtype=torch.float32) + 1).repeat(batch, 1).unsqueeze(-1).repeat(1, 1, channels)
    mask = torch.arange(LENGTH).expand(batch, LENGTH) < torch.tensor(lengths).unsqueeze(-1)
    return features * mask.unsqueeze(-1), mask


@pytest.mark.component
def test_the_crop_gathers_its_span_with_a_margin_on_each_side_and_marks_the_span():
    """A window of W gathers the positions [s - 4, s + W + 4): the span's own, marked by the indicator channel, and
    the flanking ones on each side as context, unmarked."""
    features, mask = positions()
    crop, keep = crop_window(features, mask, torch.tensor([100]), torch.tensor([130]), 48)
    assert crop.shape == (1, 48 + 2 * CROP_MARGIN, 3 + 1) and keep.shape == (1, 48 + 2 * CROP_MARGIN)
    assert crop[0, :, 0].tolist() == [float(p + 1) for p in range(96, 152)]       # positions 96 .. 151, in order
    assert crop[0, :, :3].eq(crop[0, :, :1]).all() and keep.all()
    marked = crop[0, :, 3]
    assert marked.tolist() == [0.] * 4 + [1.] * 30 + [0.] * 22                    # [100, 130) of [96, 152)
    assert CROP_WINDOWS == {'v': 320, 'd': 48, 'j': 80} and CROP_MARGIN == 4


@pytest.mark.component
def test_a_span_longer_than_its_window_is_truncated_at_its_end():
    """A V span of 400 positions in a window of 320: the window holds its first 320 + 4 positions, all marked, and
    nothing of the rest."""
    features, mask = positions()
    crop, keep = crop_window(features, mask, torch.tensor([50]), torch.tensor([450]), CROP_WINDOWS['v'])
    assert crop.shape[1] == 320 + 8 and keep.all()
    assert crop[0, 0, 0] == 46 + 1 and crop[0, -1, 0] == 373 + 1                   # positions 46 .. 373
    assert crop[0, :, 3].tolist() == [0.] * 4 + [1.] * 324


@pytest.mark.component
def test_window_positions_beyond_the_read_are_zero_and_left_out():
    """Positions before the window's array start, past the read's end and past the window array's end hold zero
    features and no mark, and are not kept: the classifier never reads them."""
    features, mask = positions(lengths=(LENGTH, 200, LENGTH))
    starts, ends = torch.tensor([2, 180, LENGTH - 30]), torch.tensor([20, 200, LENGTH])
    crop, keep = crop_window(features, mask, starts, ends, 48)
    # read 0: the window [-2, 54) starts before position 0
    assert keep[0].tolist() == [False] * 2 + [True] * 54 and crop[0, :2].eq(0).all()
    assert crop[0, 2, 0] == 0 + 1 and crop[0, 2:, 3].tolist() == [0.] * 2 + [1.] * 18 + [0.] * 34
    # read 1: 200 bases; the window [176, 232) runs 32 positions past its end
    assert keep[1].tolist() == [True] * 24 + [False] * 32 and crop[1, 24:].eq(0).all()
    assert crop[1, 23, 0] == 199 + 1 and crop[1, :24, 3].tolist() == [0.] * 4 + [1.] * 20
    # read 2: the window [542, 598) runs past the array's last position, 575
    assert keep[2].tolist() == [True] * 34 + [False] * 22 and crop[2, 34:].eq(0).all()
    assert crop[2, 33, 0] == 575 + 1


@pytest.mark.component
def test_lab_trunk_crop_v1_refuses_residue_boundaries(reference):
    from alignair.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match='residue') as caught:
        build(reference, 'lab-trunk-crop-v1', **SMALL)
    assert caught.value.context['field'] == 'boundary_mode'


@pytest.mark.component
def test_the_crop_reads_the_decoded_spans_and_no_gradient_reaches_the_boundaries(reference, monkeypatch):
    """Each gene's classifier crops the spans the product decodes (the argmax of its masked boundary logits,
    ``decoded_boundaries``), so training crops what inference crops. The allele scores' gradient reaches the trunk's
    features but not the boundary logits, nor the boundary convolution."""
    from alignair.predict.decode import decoded_boundaries
    from alignair_lab.architectures import trunk
    model = crop_model(reference).train()
    cropped = {}
    original = trunk.crop_window

    def spy(features, mask, starts, ends, window):
        cropped[window] = (starts.clone(), ends.clone())
        return original(features, mask, starts, ends, window)
    monkeypatch.setattr(trunk, 'crop_window', spy)
    features = []
    model.trunk.register_forward_hook(lambda module, inputs, output: features.append(output))
    out = model(tokens((LENGTH - 1, 380, 250)))
    for g in GENES:
        starts, ends = decoded_boundaries(out, g)
        assert torch.equal(cropped[CROP_WINDOWS[g]][0], starts) and torch.equal(cropped[CROP_WINDOWS[g]][1], ends)
    scores = sum(out[f'{g}_allele'].sum() for g in GENES)
    logits = [out[f'{g}_{edge}_logits'] for g in GENES for edge in ('start', 'end')]
    assert all(gradient is None for gradient in torch.autograd.grad(scores, logits, retain_graph=True,
                                                                    allow_unused=True))
    assert torch.autograd.grad(scores, features[0], retain_graph=True)[0].abs().sum() > 0
    scores.backward()
    assert all(model.branches[g].boundary.weight.grad is None for g in GENES)


@pytest.mark.component
def test_given_spans_are_cropped_as_given_and_the_forward_passs_own_as_decoded(reference, monkeypatch):
    """The assay's replay calls the classifier on spans of its own (``true_spans``), and on copies of the forward
    pass's expectations for the reads it has none for: a read's given span is cropped at its rounded positions, a
    read's own expectations at the spans the forward pass decoded."""
    from alignair.predict.decode import decoded_boundaries
    from alignair_lab.architectures import trunk
    model = crop_model(reference)
    batch = tokens((LENGTH - 1, 380, 250))
    with torch.no_grad():
        out = model(batch)
        features = model.trunk(model.embedding(batch['tokenized_sequence']), out['start_mask'])
    decoded = decoded_boundaries(out, 'v')
    starts, ends = out['v_start'].clone(), out['v_end'].clone()
    starts[1, 0], ends[1, 0] = 40.4, 300.6
    cropped = []
    original = trunk.crop_window

    def spy(features, mask, starts, ends, window):
        cropped.append((starts.clone(), ends.clone()))
        return original(features, mask, starts, ends, window)
    monkeypatch.setattr(trunk, 'crop_window', spy)
    with torch.no_grad():
        model.branches['v'].classify(None, starts, ends, features=features)
    assert cropped[0][0].tolist() == [int(decoded[0][0]), 40, int(decoded[0][2])]
    assert cropped[0][1].tolist() == [int(decoded[1][0]), 301, int(decoded[1][2])]


@pytest.mark.component
def test_the_cropped_classifiers_hold_a_fraction_of_conv_v1s(reference):
    """With the default options, the three cropped classifiers (a 1x1 projection, 4 dilated blocks of 128 channels,
    a mid layer and a head each) hold under a third of the parameters of lab-trunk-v2's, which are conv-v1's."""
    from alignair_lab.sizes import model_size
    crop = model_size(build(reference, 'lab-trunk-crop-v1', boundary_mode='half_open'))
    v2 = model_size(build(reference, 'lab-trunk-v2', boundary_mode='half_open'))
    assert crop['groups']['classification'] < v2['groups']['classification'] / 3
    assert (crop['groups']['segmentation'], crop['groups']['trunk']) == (v2['groups']['segmentation'],
                                                                         v2['groups']['trunk'])
    assert 'unattributed' not in crop['components']
    for g in GENES:                  # the projection and the 4 blocks' convolutions
        assert crop['components'][f'{g}.classification']['conv1d'] == 5


@pytest.mark.integration
@pytest.mark.slow
def test_lab_trunk_crop_v1_learns_one_batch(reference):
    """Updated 200 times on one batch of 8 GenAIRR reads, its V call (the top-scored allele) is one of the read's V
    alleles for more than 90% of them, though its crops start from the spans of untrained boundaries."""
    from alignair.train import overfit_one_batch
    from alignair.train.trainer import build_batch
    model = crop_model(reference)
    batch, targets = build_batch(records(8), reference, model.cfg)

    def v_top1():
        with torch.no_grad():
            called = model.eval()(batch)['v_allele'].argmax(-1)
        return float(targets['v_allele'].gather(1, called.unsqueeze(-1)).gt(0).float().mean())

    before = v_top1()
    overfit_one_batch(model, batch, targets, steps=200, lr=1e-3)
    assert v_top1() > 0.9 > before


# --- lab-trunk-meta-v1 and lab-trunk-crop-meta-v1: the metadata heads read the trunk ------------------------------

@pytest.mark.component
@pytest.mark.parametrize('architecture', ['lab-trunk-meta-v1', 'lab-trunk-crop-meta-v1'])
def test_the_trunk_meta_tower_pools_the_reads_trunk_features(reference, architecture):
    """The metadata heads' input is the read's masked mean and max trunk features through Linear(2C, block_out) and
    GELU: no convolution of its own, and nothing from the padding."""
    model = build(reference, architecture, **SMALL, boundary_mode='half_open')
    tower = model.meta_tower
    assert count(tower) == 2 * TRUNK_CHANNELS * model.cfg.block_out + model.cfg.block_out
    assert not any(isinstance(module, torch.nn.Conv1d) for module in tower.modules())
    features, seen = [], []
    model.trunk.register_forward_hook(lambda module, inputs, output: features.append(output))
    tower.register_forward_hook(lambda module, inputs, output: seen.append(output))
    batch = tokens()
    with torch.no_grad():
        model(batch)
    keep = (batch['tokenized_sequence'] != 0).unsqueeze(-1)
    pooled = torch.cat([(features[0] * keep).sum(1) / keep.sum(1),
                        features[0].masked_fill(~keep, float('-inf')).amax(1)], -1)
    assert torch.allclose(seen[0], torch.nn.functional.gelu(tower.proj(pooled)), atol=1e-5)
    padded = dict(batch)
    padded['tokenized_sequence'] = batch['tokenized_sequence'][1:2]       # the 380-long read alone
    with torch.no_grad():
        model(padded)
    assert torch.allclose(seen[1], seen[0][1:2], atol=1e-5)


@pytest.mark.component
def test_lab_trunk_crop_meta_v1_has_no_meta_conv_tower(reference):
    """With the default options, the trunk meta tower replaces conv-v1's meta conv tower: the meta group shrinks to
    the projection and the heads, and the model holds no Conv1d outside the trunk, the boundaries and the crops."""
    from alignair_lab.sizes import model_size
    crop = model_size(build(reference, 'lab-trunk-crop-v1', boundary_mode='half_open'))
    meta = model_size(build(reference, 'lab-trunk-crop-meta-v1', boundary_mode='half_open'))
    assert meta['components']['meta_tower']['conv1d'] == 0
    assert meta['groups']['meta'] < crop['groups']['meta']
    assert meta['conv1d'] == crop['conv1d'] - crop['components']['meta_tower']['conv1d']
    assert (meta['groups']['trunk'], meta['groups']['classification']) == (crop['groups']['trunk'],
                                                                           crop['groups']['classification'])


@pytest.mark.integration
@pytest.mark.slow
def test_lab_trunk_meta_v1_learns_one_batch(reference):
    """Updated 200 times on one batch of 8 GenAIRR reads, its mutation-rate error falls by more than half."""
    from alignair.train import overfit_one_batch
    from alignair.train.trainer import build_batch
    model = build(reference, 'lab-trunk-meta-v1', **SMALL, boundary_mode='half_open')
    batch, targets = build_batch(records(8), reference, model.cfg)

    def error():
        with torch.no_grad():
            rate = model.eval()(batch)['mutation_rate'].reshape(-1)
        return float((rate - targets['mutation_rate'].reshape(-1)).abs().mean())

    before = error()
    overfit_one_batch(model, batch, targets, steps=200, lr=1e-3)
    assert error() < before / 2


# --- the model file, the replay and the screen -------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize('architecture,options', [
    ('lab-trunk-v1', {}), ('lab-trunk-v2', {'boundary_mode': 'half_open'}),
    ('lab-trunk-v2', {'boundary_mode': 'half_open', 'region_head': True, 'anchor_heads': True}),
    ('lab-trunk-crop-v1', {'boundary_mode': 'half_open', 'anchor_heads': True, 'd_presence_head': True})],
    ids=['v1', 'v2', 'v2-regions-anchors', 'crop-anchors-d-presence'])
def test_a_lab_trunk_model_file_round_trips(reference, tmp_path, monkeypatch, architecture, options):
    """Written as a training run writes its model.alignair, it loads back (in a process that registered the lab's
    candidates) and predicts what it predicted, the junction fields included (from the anchors, with anchor heads);
    a process that did not is refused, naming the architecture."""
    from alignair.core import components
    from alignair.errors import ConfigurationError
    from alignair.train.checkpoints import save_checkpoint
    model = build(reference, architecture, **SMALL, **options)
    path = tmp_path / 'model.alignair'
    save_checkpoint(path, model.cfg, model, None, 7, dataconfigs=[gd.HUMAN_IGH_OGRDB])
    loaded, loaded_reference = api.load_model(str(path))
    assert loaded.cfg == model.cfg and loaded.cfg.architecture == architecture
    assert loaded.cfg.boundary_mode == options.get('boundary_mode', 'residue')
    assert list(loaded.state_dict()) == list(model.state_dict())
    with torch.no_grad():
        first, second = model(tokens()), loaded(tokens())
    assert all(torch.equal(first[key], second[key]) for key in first)
    sequences = [record['sequence'] for record in records(3, seed=11)]
    rows = api.predict_sequences(model, reference, sequences, device='cpu')
    assert api.predict_sequences(loaded, loaded_reference, sequences, device='cpu') == rows
    assert loaded.cfg.anchor_heads == options.get('anchor_heads', False)
    assert all(('v_anchor' in row) == loaded.cfg.anchor_heads for row in rows)
    assert all('junction' in row and 'productive' in row for row in rows)
    monkeypatch.setattr(components, '_ARCHITECTURES',
                        {name: value for name, value in components._ARCHITECTURES.items() if name != architecture})
    with pytest.raises(ConfigurationError, match=architecture):
        api.load_model(str(path))


@pytest.mark.integration
def test_the_assay_scores_a_d_presence_model_on_its_gated_rows(reference):
    """On a model with a D presence head the assay's replay is the product's prediction, gated D included, its head
    metrics run, and every ablation variant runs: each row carries the head's class, and only a callable D keeps a
    call. A D the head calls absent has no coordinates."""
    from alignair_assay.core.reference import Reference
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS, run_variants
    from alignair_assay.diagnostics.heads import head_scores
    from alignair_assay.diagnostics.replay import replay
    from alignair_assay.truth.build import truth_from_genairr_record
    model = build(reference, 'lab-trunk-v2', boundary_mode='half_open', **SMALL, d_presence_head=True)
    truth_reference = Reference.from_dataconfig('HUMAN_IGH_OGRDB')
    found = records(4, seed=13)
    cases = [{'case_id': f'c{k}', 'sequence': record['sequence'].upper()} for k, record in enumerate(found)]
    truths = [truth_from_genairr_record(record, truth_reference, case_id=case['case_id'], stratum='s')
              for record, case in zip(found, cases)]
    sequences, ids = [case['sequence'] for case in cases], [case['case_id'] for case in cases]
    outputs = replay(model, reference, sequences, ids, device='cpu', batch_size=2)
    assert outputs.airr_rows == api.predict_sequences(model, reference, sequences, device='cpu', batch_size=2)
    assert head_scores(outputs, truths)
    results = run_variants(model, reference, cases, truths, reference=truth_reference, device='cpu', batch_size=2,
                           variants=DEFAULT_VARIANTS)
    assert list(results) == list(DEFAULT_VARIANTS)
    for result in results.values():
        for row in result.rows:
            assert row['d_presence'] in ('absent', 'short', 'callable')
            assert bool(row['d_call']) <= (row['d_presence'] == 'callable')
            assert row['d_presence'] != 'absent' or row.get('d_sequence_start') is None


@pytest.mark.integration
def test_the_assay_scores_an_anchor_model_from_its_anchors(reference):
    """On a model with anchor heads the assay's replay is the product's prediction, its head metrics run without a
    productive head (head.productive.correct is not scored), and every ablation variant runs: each row's junction and
    flags are the anchors' derivation over the row's own V start and J end, and the flags are scored."""
    from alignair.predict.junction import derive_junction
    from alignair_assay.core.reference import Reference
    from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS, run_variants
    from alignair_assay.diagnostics.heads import head_scores
    from alignair_assay.diagnostics.replay import replay
    from alignair_assay.truth.build import truth_from_genairr_record
    model = build(reference, 'lab-trunk-v2', boundary_mode='half_open', **SMALL, anchor_heads=True)
    truth_reference = Reference.from_dataconfig('HUMAN_IGH_OGRDB')
    found = records(4, seed=13)
    cases = [{'case_id': f'c{k}', 'sequence': record['sequence'].upper()} for k, record in enumerate(found)]
    truths = [truth_from_genairr_record(record, truth_reference, case_id=case['case_id'], stratum='s')
              for record, case in zip(found, cases)]
    sequences, ids = [case['sequence'] for case in cases], [case['case_id'] for case in cases]
    outputs = replay(model, reference, sequences, ids, device='cpu', batch_size=2)
    assert outputs.airr_rows == api.predict_sequences(model, reference, sequences, device='cpu', batch_size=2)
    heads = head_scores(outputs, truths)
    assert heads and all('head.productive.correct' not in case and 'head.orientation.correct' in case
                         for case in heads.values())
    results = run_variants(model, reference, cases, truths, reference=truth_reference, device='cpu', batch_size=2,
                           variants=DEFAULT_VARIANTS)
    assert list(results) == list(DEFAULT_VARIANTS)
    for result in results.values():
        for row in result.rows:
            call = derive_junction(row['sequence'], row['v_anchor'], row['j_anchor'],
                                   coding_span=(row.get('v_sequence_start'), row.get('j_sequence_end')))
            assert (row['junction_start'], row['productive'], row['productive_prediction']) == (
                call.junction_start, call.productive, call.productive)
        assert all('productive.correct' in case for case in result.scores.values())


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
    """The screen trains the candidate, then the assay diagnoses its model file in the screen's process. Its record and
    headline give both models' sizes: the candidate's with its trunk, the baseline's without."""
    from alignair.model_file import read_model_state
    from alignair_lab.cli import main
    assert main(candidate_arguments(rig, 'trunk', architecture='lab-trunk-v1')) == 0
    record = json.loads((rig.screens / 'trunk' / 'screen.json').read_text())
    assert (record['status'], record['architecture'], record['baseline']['name']) == ('done', 'lab-trunk-v1', 'base0')
    assert read_model_state(record['candidate']['model_path']).config.architecture == 'lab-trunk-v1'
    assert set(record['comparison']['suites']) == {'S0'} and record['candidate']['model_key']
    candidate, base = record['candidate']['size'], record['baseline']['size']
    assert (candidate['architecture'], base['architecture']) == ('lab-trunk-v1', 'lab-null-v1')
    assert candidate['components']['trunk']['parameters'] > 0 and 'trunk' not in base['components']
    assert candidate['conv1d'] < base['conv1d']
    headline = (rig.screens / 'trunk' / 'headline.txt').read_text()
    assert 'Size of the candidate (lab-trunk-v1): ' in flat(headline)
    assert 'Size of the baseline (lab-null-v1): ' in flat(headline)
