"""Independent source-order, receptor-policy, injection, and diagnostic contracts."""
from itertools import islice
from types import SimpleNamespace, ModuleType
import sys

import pytest

from alignair.errors import ConfigurationError, TrainingDataError, current_error_context
from alignair.train.gym import streams
from alignair.train.gym.curriculum.base import INDEL_FRACTION
from alignair.train.gym.streams import (SEED_BLOCK_RECORDS, SELECTED_FRACTION, CurriculumStreamPolicy,
                                       dataconfig_name, effective_mutation_caps, seed_block, training_records,
                                       stream_records)

pytestmark = pytest.mark.component


def reference(name='IGH', chain_type='BCR_HEAVY'):
    return SimpleNamespace(name=name, metadata=SimpleNamespace(chain_type=chain_type))


def provider(dataconfig, parameters, seed):
    for ordinal in range(3):
        yield {'panel': dataconfig.name, 'seed': seed, 'ordinal': ordinal, 'rate': parameters['mutation_rate']}


def test_auto_single_source_and_explicit_amplicon_mixture_have_distinct_contracts():
    automatic = CurriculumStreamPolicy((.3,), 0.)
    assert len(automatic.parameters_for(reference())) == 1
    assert automatic.parameters_for(reference())[0]['mutation_rate'] == pytest.approx(.0485)
    mixture = CurriculumStreamPolicy((.3,), 0., mode='amplicon')
    params = mixture.parameters_for(reference())
    assert len(params) == 4
    assert [(row['end_loss_5'], row['end_loss_3']) for row in params[1:]] == [
        ((0,15),(150,350)), ((150,350),(0,15)), ((70,175),(70,175))]
    assert len(CurriculumStreamPolicy((), 0., mode='amplicon').parameters_for(reference())) == 3


def test_default_mix_repeats_profiles_independently_and_caps_each_reference():
    policy = CurriculumStreamPolicy(short_boost=2, mutation_cap=.1)
    igh = policy.parameters_for(reference())
    tcr = policy.parameters_for(reference('TRB','TCR_BETA'))
    assert len(igh) == 10 and max(row['mutation_rate'] for row in igh) == .1
    assert {row['mutation_rate'] for row in tcr} == {0.}
    igh[3]['mutation_rate'] = .9
    assert igh[6]['mutation_rate'] != .9
    assert max(row['mutation_rate'] for row in policy.parameters_for(reference())) == .1


UNSELECTED = CurriculumStreamPolicy((.3,), 0., mode='amplicon', selected_fraction=0)   # one stream per component


def seeds_of(references=(reference(),), policy=CurriculumStreamPolicy((.3,),0.,mode='amplicon'), **options):
    """The GenAIRR seed of every provider call of one stream, in call order."""
    calls = []
    def observed(dc, params, seed):
        calls.append(seed)
        return provider(dc, params, seed)
    next(training_records(references, policy=policy, record_source=observed, **options))
    return calls


def blocks_of(policy, seed, *, panel=0, purpose='training', start_step=0, force_mixture=False):
    """The seed blocks of each component's streams, in their interleaving order."""
    return [[seed_block(seed, purpose, start_step, panel, component, name) for name, _ in streams]
            for component, streams in enumerate(policy.substreams_for(reference(), force_mixture=force_mixture))]


def test_components_use_their_own_seed_blocks_and_round_robin_order_without_eager_reads():
    calls = []
    def observed(dc, params, seed):
        calls.append(seed)
        yield from provider(dc,params,seed)
    records = training_records((reference(),), policy=UNSELECTED, seed=17, record_source=observed)
    assert calls == []
    first = next(records)
    blocks = [streams[0] for streams in blocks_of(UNSELECTED, 17)]
    assert calls == blocks and first['seed'] == blocks[0]
    rest = list(records)
    assert [(row['seed'],row['ordinal']) for row in [first,*rest]] == [
        (seed,ordinal) for ordinal in range(3) for seed in blocks]


def test_a_component_interleaves_three_selected_streams_with_one_unselected_stream():
    calls = []
    def observed(dc, params, seed):
        calls.append((seed, params.get('productive_only', False)))
        yield from provider(dc, params, seed)
    policy = CurriculumStreamPolicy((.3,), 0., mode='amplicon')   # p=.3, V-anchored, J-anchored, fragment amplicons
    assert policy.selected_fraction == SELECTED_FRACTION == .75
    assert policy.selection_pattern == (True, True, True, False)
    rows = list(training_records((reference(),), policy=policy, seed=17, record_source=observed))
    blocks = blocks_of(policy, 17)
    assert [seed for seed, _ in calls] == [block for streams in blocks for block in streams]
    assert len({seed for seed, _ in calls}) == 16
    assert [selected for _, selected in calls] == [True, True, True, False, *[False] * 12]
    # record k of a component comes from its stream k % 4; the components take turns record by record
    assert [(row['seed'], row['ordinal']) for row in rows] == [
        (blocks[component][k % 4], k // 4) for k in range(12) for component in range(4)]


def test_only_full_length_components_are_selected_and_every_amplicon_stays_unselected():
    policy = CurriculumStreamPolicy()                              # the 1.0.0 mix
    assert policy.selected_shares() == (.75, .75, .75, 0., 0., 0., .75)
    assert CurriculumStreamPolicy(short_boost=2, selected_fraction=.5).selected_shares() == (
        .5, .5, .5, 0., 0., 0., 0., 0., 0., .5)
    components = policy.parameters_for(reference())
    streams = policy.substreams_for(reference())
    assert [name for name, _ in streams[0]] == ['selected-0', 'selected-1', 'selected-2', 'unselected-0']
    assert [name for name, _ in streams[3]] == [name for name, _ in streams[4]] == [name for name, _ in streams[5]] == [
        'unselected-0', 'unselected-1', 'unselected-2', 'unselected-3']
    for component, substreams in zip(components, streams):
        for name, parameters in substreams:
            assert parameters == ({**component, 'productive_only': True} if name.startswith('selected') else component)


@pytest.mark.parametrize('fraction,pattern', [
    (0, (False,)), (1, (True,)), (.5, (True, False)), (.75, (True, True, True, False)),
    (.4, (True, False, True, False, False)), (1 / 3, (True, False, False)),
    (.7, (True, True, True, False, True, True, False, True, True, False))])
def test_selected_records_are_spread_evenly_over_each_period(fraction, pattern):
    policy = CurriculumStreamPolicy(selected_fraction=fraction)
    assert policy.selection_pattern == pattern and sum(pattern) / len(pattern) == pytest.approx(fraction)


def test_a_streams_seed_block_is_a_reserved_sha256_derived_block_of_2_32_records():
    import hashlib, json
    key = json.dumps(['alignair.training_records', 0, 'training', 0, 0, 0, 'selected-0']).encode()
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 2**28
    assert SEED_BLOCK_RECORDS == 2**32
    assert seed_block(0, 'training', 0, 0, 0, 'selected-0') == 2**62 + index * 2**32
    for purpose in ('training', 'validation', 'probe'):
        block = seed_block(5, purpose, 3, 1, 2, 'unselected-0')
        assert 2**62 <= block and block + SEED_BLOCK_RECORDS <= 2**63 and block % SEED_BLOCK_RECORDS == 0
        assert block == seed_block(5, purpose, 3, 1, 2, 'unselected-0')


def test_streams_of_a_run_never_share_a_block_or_reach_the_assays_seeds():
    policy = CurriculumStreamPolicy((.3, .6, .9), .25, 1)       # the 1.0.0 mix: 7 components of 4 streams per panel
    panels = (reference(), reference('TRA', 'TCR_ALPHA'))
    for seed in range(20):
        training = seeds_of(panels, policy, seed=seed)
        resumed = seeds_of(panels, policy, seed=seed, start_step=1000)
        validation = seeds_of(panels, policy, seed=seed, purpose='validation')
        probe = seeds_of(panels, policy, seed=seed, purpose='probe')
        blocks = [*training, *resumed, *validation, *probe]
        assert len(training) == 56 and len(set(blocks)) == len(blocks)
        assert all(block >= 2**62 and block % SEED_BLOCK_RECORDS == 0 for block in blocks)   # far above [0, 2**31)
    assert not set(seeds_of(seed=0)) & set(seeds_of(seed=1))
    regions = {purpose: {block >> 60 for seed in range(50) for block in seeds_of(seed=seed, purpose=purpose)}
               for purpose in ('training', 'validation', 'probe')}
    assert regions == {'training': {4}, 'validation': {5}, 'probe': {6}}   # each purpose its own range of blocks


def test_two_streams_drawing_the_same_block_are_refused(monkeypatch):
    monkeypatch.setattr(streams, 'seed_block', lambda *key: 2**62)
    with pytest.raises(ConfigurationError, match='same seed block'):
        next(training_records((reference(),), policy=CurriculumStreamPolicy((.3,),0.,mode='amplicon'),
                              record_source=provider))


def test_finite_components_emit_complete_rounds_with_the_historical_partial_round_consumption():
    pulled = []
    blocks = [streams[0] for streams in blocks_of(UNSELECTED, 0)]
    def finite(dc,params,seed):
        for ordinal in range(1 if seed==blocks[1] else 2):
            pulled.append((seed,ordinal))
            yield {'seed':seed,'ordinal':ordinal}
    rows=list(training_records((reference(),),policy=UNSELECTED, record_source=finite))
    assert len(rows)==4
    assert pulled==[(blocks[0],0),(blocks[1],0),(blocks[2],0),(blocks[3],0),(blocks[0],1)]


def test_a_component_ends_with_its_last_complete_period():
    pulled = []
    policy = CurriculumStreamPolicy((.3,), 0.)                    # one component of 4 streams
    blocks = blocks_of(policy, 0)[0]
    def finite(dc, params, seed):
        for ordinal in range(1 if seed == blocks[2] else 3):
            pulled.append((seed, ordinal))
            yield {'seed': seed, 'ordinal': ordinal}
    rows = list(training_records((reference(),), policy=policy, record_source=finite))
    assert [(row['seed'], row['ordinal']) for row in rows] == [(block, 0) for block in blocks]
    assert pulled == [*((block, 0) for block in blocks), (blocks[0], 1), (blocks[1], 1)]


def test_multiple_panels_use_independent_seed_blocks_and_distinct_locus_class_ids():
    policy = CurriculumStreamPolicy((.3,), 0., selected_fraction=0)
    rows=list(islice(training_records((reference(),reference('TRA','TCR_ALPHA'),reference('IGH2')),
        policy=policy,seed=17,record_source=provider),12))
    assert [row['chain_type'] for row in rows]==[0,1,0]*4
    blocks = [blocks_of(policy, 17, panel=panel, force_mixture=True) for panel in range(3)]
    assert [row['seed'] for row in rows]==[blocks[panel][component][0] for component in range(4) for panel in range(3)]
    assert {row['rate'] for row in rows if row['panel']=='TRA'}=={0.}
    assert all(row['rate']>0 for row in rows if row['panel']!='TRA')


def test_locus_tagging_does_not_mutate_or_relabel_the_providers_own_record():
    original={'source':1}
    def shared(*args):yield original
    rows=list(training_records((reference(),reference('TRB','TCR_BETA')),
         policy=CurriculumStreamPolicy((.3,),0.),record_source=shared))
    assert {row['chain_type'] for row in rows}=={0,1}
    assert original=={'source':1} and all(row is not original for row in rows)


def test_translation_filter_only_checks_translation_eligibility():
    rows=[{'sequence':'ATG','sequence_aa':aa,'junction_start':frame,'productive':False}
          for aa,frame in [('',0),('M',None),('MM',0),('M',0),('*',0)]]
    def source(*args):return iter(rows)
    actual=list(training_records((reference(),),policy=CurriculumStreamPolicy((.3,),0.,selected_fraction=0),
                                  translated_only=True,record_source=source))
    assert [row['sequence_aa'] for row in actual]==['M','*']
    assert all(not row['productive'] for row in actual)


def test_filter_errors_carry_source_context_without_leaking_to_the_runner():
    def invalid(*args):yield {'sequence':'ATG','sequence_aa':'M','junction_start':'bad'}
    records=training_records((reference(),),policy=CurriculumStreamPolicy((.3,),0.),
                              seed=19,translated_only=True,record_source=invalid)
    with pytest.raises(TrainingDataError) as caught:next(records)
    assert caught.value.context['stage']=='curriculum_source' and caught.value.context['seed']==19
    assert current_error_context()=={}


def test_source_factory_failure_preserves_type_and_attempted_seed():
    def fail(*args):raise TrainingDataError('provider failed')
    records=training_records((reference(),),record_source=fail,seed=9)
    with pytest.raises(TrainingDataError) as caught:next(records)
    assert caught.value.context['seed']==9 and current_error_context()=={}


def test_default_provider_copies_parameters_and_releases_its_context(monkeypatch):
    seen=[]
    def build(dc,params,*,allow_curatable):
        assert allow_curatable and params['invert_d_prob']==0.
        params['mutation_rate']=.9
        def records(n,seed):
            seen.append((n,seed))
            yield {'seed':seed}
            raise TrainingDataError('read failed')
        return SimpleNamespace(stream_records=records)
    monkeypatch.setattr(streams,'build_experiment',build)
    params={'mutation_rate':.1}
    records=stream_records(reference(),params,13)
    assert seen==[] and next(records)=={'seed':13}
    assert params=={'mutation_rate':.1} and current_error_context()=={}
    with pytest.raises(TrainingDataError) as caught:next(records)
    assert caught.value.context['stage']=='curriculum_component' and caught.value.context['seed']==13


def test_reference_name_resolution_is_lazy_and_reports_unknown_names(monkeypatch):
    module=ModuleType('GenAIRR')
    module.KNOWN=reference()
    monkeypatch.setitem(sys.modules,'GenAIRR',module)
    records=training_records(('KNOWN',),policy=CurriculumStreamPolicy((.3,),0.),record_source=provider)
    assert next(records)['panel']=='IGH'
    unknown=training_records(('UNKNOWN',),record_source=provider,seed=3)
    with pytest.raises(ConfigurationError) as caught:next(unknown)
    assert caught.value.context['seed']==3
    with pytest.raises(ConfigurationError):next(training_records((object(),),record_source=provider))
    assert current_error_context()=={}


def test_explicit_sources_and_custom_names_work_without_importing_the_simulator(monkeypatch):
    import builtins
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.startswith('GenAIRR'):raise ImportError('not installed')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    assert dataconfig_name(reference('custom'))=='custom'
    assert dataconfig_name(reference('unnamed','TCR_BETA'))=='TCR_BETA'
    assert dataconfig_name(object())=='unknown'
    assert next(training_records((reference(),),record_source=provider))['panel']=='IGH'


def test_materialized_provider_names_and_custom_name_collisions_are_preserved(monkeypatch):
    item=reference('custom')
    module=ModuleType('GenAIRR.data')
    module._private=item
    module.CANONICAL=item
    package=ModuleType('GenAIRR')
    package.__path__=[]
    package.data=module
    monkeypatch.setitem(sys.modules,'GenAIRR',package)
    monkeypatch.setitem(sys.modules,'GenAIRR.data',module)
    assert dataconfig_name(item)=='CANONICAL'
    caps=effective_mutation_caps((reference('x'),reference('x#2'),reference('x','TCR_BETA')),None)
    assert caps=={'x':None,'x#2':None,'x#3':0.}


@pytest.mark.parametrize('options,field',[
    ({'mode':'random'},'mode'),({'progresses':None},'progresses'),({'progresses':'0.3'},'progresses'),
    ({'progresses':()},'progresses'),({'progresses':(float('nan'),)},'progresses'),
    ({'heavy_shm':True},'heavy_shm'),({'heavy_shm':1.1},'heavy_shm'),
    ({'short_boost':0},'short_boost'),({'short_boost':False},'short_boost'),
    ({'short_boost':1.5},'short_boost'),({'mutation_cap':-.1},'mutation_cap'),
    ({'indel_fraction':None},'indel_fraction'),({'indel_fraction':(.3,.1)},'indel_fraction'),
    ({'indel_form':'realistic'},'indel_form'),({'indel_form':None},'indel_form'),
    ({'selected_fraction':1.1},'selected_fraction'),({'selected_fraction':-.25},'selected_fraction'),
    ({'selected_fraction':True},'selected_fraction'),({'selected_fraction':float('nan')},'selected_fraction'),
    ({'selected_fraction':'0.75'},'selected_fraction'),({'selected_fraction':.123},'selected_fraction'),
])
def test_invalid_policy_is_rejected_before_source_construction(options,field):
    with pytest.raises(ConfigurationError) as caught:CurriculumStreamPolicy(**options)
    assert caught.value.context['field']==field


def test_the_historical_uniform_indel_form_reproduces_the_1_0_0_component_parameters():
    policy = CurriculumStreamPolicy(indel_form='uniform')
    # progresses .3/.6/.9, then the three amplicon profiles at .6 and heavy SHM at .3
    assert [component['indel_count'] for component in policy.parameters_for(reference())] == [
        (0, 2), (0, 3), (0, 4), (0, 3), (0, 3), (0, 3), (0, 2)]


def test_every_component_draws_indels_from_the_policys_indel_fraction_ramp():
    assert CurriculumStreamPolicy().indel_form == 'weighted'
    assert CurriculumStreamPolicy().indel_fraction == INDEL_FRACTION
    policy = CurriculumStreamPolicy(indel_fraction=[0, 1])
    assert policy.indel_fraction == (0., 1.)
    # progresses .3/.6/.9, then the three amplicon profiles at .6 and heavy SHM at .3
    shares = [1 - dict(component['indel_count'])[0] for component in policy.parameters_for(reference())]
    assert shares == pytest.approx([.3, .6, .9, .6, .6, .6, .3])
    for component in policy.parameters_for(reference()):
        assert [count for count, _ in component['indel_count']] == [0, 1, 2, 3]


@pytest.mark.parametrize('references,options',[
    ((),{}),('KNOWN',{}),(None,{}),((reference(),),{'policy':None}),
    ((reference(),),{'seed':-1}),((reference(),),{'seed':True}),
    ((reference(),),{'translated_only':1}),((reference(),),{'record_source':1}),
    ((reference(),),{'purpose':'testing'}),((reference(),),{'purpose':None}),
    ((reference(),),{'start_step':-1}),((reference(),),{'start_step':1.5}),
])
def test_invalid_source_request_is_rejected_eagerly(references,options):
    with pytest.raises(ConfigurationError):training_records(references,**options)


def test_receptor_policy_uses_enum_values_and_matches_experiment_detection():
    from enum import Enum
    from alignair.reference.reference_set import is_tcr
    class Kind(Enum):
        BETA='TCR_BETA'
        HEAVY='BCR_HEAVY'
    assert is_tcr(reference(chain_type=Kind.BETA))
    assert not is_tcr(reference(chain_type=Kind.HEAVY))
    assert {row['mutation_rate'] for row in CurriculumStreamPolicy().parameters_for(reference(chain_type=Kind.BETA))}=={0.}
