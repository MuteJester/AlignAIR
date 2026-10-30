"""Independent source-order, receptor-policy, injection, and diagnostic contracts."""
from itertools import islice
from types import SimpleNamespace, ModuleType
import sys

import pytest

from alignair.errors import ConfigurationError, TrainingDataError, current_error_context
from alignair.train.gym import streams
from alignair.train.gym.curriculum.base import INDEL_FRACTION
from alignair.train.gym.streams import (CurriculumStreamPolicy, dataconfig_name, effective_mutation_caps,
                                       training_records, stream_records)

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


def test_components_use_consecutive_seeds_and_round_robin_order_without_eager_reads():
    calls = []
    def observed(dc, params, seed):
        calls.append(seed)
        yield from provider(dc,params,seed)
    records = training_records((reference(),), policy=CurriculumStreamPolicy((.3,),0.,mode='amplicon'),
                               seed=17,record_source=observed)
    assert calls == []
    first = next(records)
    assert calls == [17,18,19,20] and first['seed'] == 17
    rest = list(records)
    assert [(row['seed'],row['ordinal']) for row in [first,*rest]] == [
        (seed,ordinal) for ordinal in range(3) for seed in (17,18,19,20)]


def test_finite_components_emit_complete_rounds_with_the_historical_partial_round_consumption():
    pulled = []
    def finite(dc,params,seed):
        for ordinal in range(1 if seed==1 else 2):
            pulled.append((seed,ordinal))
            yield {'seed':seed,'ordinal':ordinal}
    rows=list(training_records((reference(),),policy=CurriculumStreamPolicy((.3,),0.,mode='amplicon'),
                               record_source=finite))
    assert len(rows)==4
    assert pulled==[(0,0),(1,0),(2,0),(3,0),(0,1)]


def test_multiple_panels_use_independent_seed_blocks_and_distinct_locus_class_ids():
    rows=list(islice(training_records((reference(),reference('TRA','TCR_ALPHA'),reference('IGH2')),
        policy=CurriculumStreamPolicy((.3,),0.),seed=17,record_source=provider),12))
    assert [row['chain_type'] for row in rows]==[0,1,0]*4
    assert [row['seed'] for row in rows]==[17,1030,2043,18,1031,2044,19,1032,2045,20,1033,2046]
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
    actual=list(training_records((reference(),),policy=CurriculumStreamPolicy((.3,),0.),
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
