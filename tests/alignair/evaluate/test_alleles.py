import pytest
import torch
from dataclasses import replace
from types import SimpleNamespace

from alignair.core import AlignAIRConfig
from alignair.errors import ConfigurationError, NumericalError, TrainingDataError, current_error_context
from alignair.evaluate import collect_alignment_observations, evaluate_allele_cohort
from alignair.evaluate.cohort import ClassificationObservation, PredictionSetObservation, allele_observations
from alignair.evaluate.alleles import (allele_observation_counts, empty_allele_counts, finalize_gene,
                                      gene_allele_counts, merge_allele_counts)

pytestmark = pytest.mark.component


def test_within_class_synonymous_confusion_is_detected():
    # 3 alleles; a0 and a1 are synonymous (same protein class 0), a2 is class 1.
    names = ["a0", "a1", "a2"]
    class_of = {"a0": 0, "a1": 0, "a2": 1}
    # read 0: truth a0; model spreads 0.9/0.8 over the synonymous pair, 0.1 on a2 -> argmax a0 (right nt)
    # read 1: truth a0; model argmax a1 (WRONG nt but SAME class), true a0 still 0.7 (in set)
    probs = torch.tensor([[0.9, 0.8, 0.1],
                          [0.7, 0.95, 0.1]])
    target = torch.tensor([[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0]])
    c = gene_allele_counts(probs, target, names, class_of, threshold=0.5)
    assert c["n"] == 2
    assert c["nt_top1"] == 1                    # only read 0 gets nt argmax right
    assert c["class_top1"] == 2                 # BOTH get the protein class right
    assert c["errors"] == 1                     # read 1 is an nt error...
    assert c["errors_within_class"] == 1        # ...but a synonymous (within-class) one
    assert c["true_in_predicted_set"] == 2      # true allele > 0.5 in both reads
    assert c["set_size"] == 4                   # read0 {a0,a1}=2, read1 {a0,a1}=2


def test_finalize_computes_the_emergent_ambiguity_signal():
    counts = {"n": 100, "nt_top1": 70, "class_top1": 98, "errors": 30,
              "errors_within_class": 29, "true_in_predicted_set": 96, "set_size": 180}
    r = finalize_gene(counts, has_class=True)
    assert r["nt_top1"] == 0.70 and r["class_top1"] == 0.98
    assert abs(r["frac_errors_within_class"] - 29 / 30) < 1e-9   # ~all nt errors are synonymous
    assert r["true_in_predicted_set"] == 0.96
    assert r["mean_pred_set_size"] == 1.8


def test_unsupervised_rows_are_skipped():
    # an all-zero target row (e.g. inverted D) contributes nothing
    probs = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    c = gene_allele_counts(probs, target, ["a", "b"], None, threshold=0.5)
    assert c["n"] == 1


def test_missing_class_entries_never_count_as_a_none_to_none_match():
    counts = gene_allele_counts(torch.tensor([[.2,.8],[.9,.1],[.1,.9]]),
                               torch.tensor([[1.,0.],[1.,0.],[0.,1.]]), ['a','b'], {})
    assert counts['n'] == 3 and counts['nt_top1'] == 2
    assert counts['class_support'] == counts['class_top1'] == 0
    report = finalize_gene(counts, True)
    assert report['class_top1'] == 0.
    assert report['unmapped_class_rows'] == 3


def test_partial_maps_have_independent_class_and_error_denominators():
    names = ['a','b','c']
    counts = gene_allele_counts(torch.tensor([[.1,.9,0.],[.1,.8,.1],[0.,.1,.9]]),
                               torch.tensor([[1.,0.,0.],[0.,1.,0.],[1.,0.,0.]]),
                               names, {'a':0,'b':0})
    report = finalize_gene(counts, True)
    assert report['n'] == 3 and report['nt_top1'] == pytest.approx(1/3)
    assert report['nt_error_rate'] == pytest.approx(2/3)
    assert report['class_support'] == 2 and report['class_top1'] == 1.
    assert report['class_error_support'] == 1 and report['frac_errors_within_class'] == 1.
    assert report['unmapped_class_rows'] == 1
    # Even a partly mapped truth set has no fully defined equivalence comparison.
    counts = gene_allele_counts(torch.tensor([[.8,.1,.1]]), torch.tensor([[1.,0.,1.]]), names, {'a':0,'b':0})
    assert counts['nt_top1'] == 1 and counts['class_support'] == 0


@pytest.mark.parametrize('threshold, expected', [(0.,(0,1)), (.5,(1,)), (1.,())])
def test_thresholded_sets_are_strictly_greater_than_the_declared_threshold(threshold, expected):
    rows, sets = allele_observations(torch.tensor([[.5,.8]]), torch.tensor([[1.,0.]]),
                                    prediction_set_threshold=threshold, record_offset=7)
    assert rows[0].record_index == sets[0].record_index == 7
    assert sets[0].members == expected


@pytest.mark.parametrize('threshold', [-.1, 1.1, True, float('nan'), float('inf'), '0.5'])
def test_invalid_prediction_set_thresholds_fail_before_collection(threshold):
    with pytest.raises(ConfigurationError):
        allele_observations(torch.ones(1,1), torch.ones(1,1), prediction_set_threshold=threshold)


@pytest.mark.parametrize('probabilities', [None, torch.ones(2), torch.ones(1,0), torch.ones(1,2,dtype=torch.long)])
def test_invalid_probability_matrix_is_classified(probabilities):
    with pytest.raises(TrainingDataError):
        allele_observations(probabilities, None)


@pytest.mark.parametrize('names, classes', [(None,None), ('ab',None), (['a','a'],None),
                                         ([''],None), (['a'],[]), (['a'],{'a':[]}),
                                         (['a'],{'a':-1}), (['a'],{'a':float('nan')}), (['a'],{'a':True})])
def test_invalid_vocabulary_or_class_policy_is_rejected(names, classes):
    with pytest.raises(ConfigurationError):
        allele_observation_counts([], [], names, classes)


def test_counts_validate_record_alignment_and_vocabulary_indices():
    row = ClassificationObservation(0,(0,),1)
    predicted = PredictionSetObservation(0,(1,))
    for rows, sets in [([row],[]), ([row],[predicted._replace(record_index=1)]),
                       ([row._replace(truth=())],[predicted]), ([None],[predicted]),
                       ([row._replace(prediction=2)],[predicted]),
                       ([row._replace(prediction=True)],[predicted]),
                       ([row._replace(truth=('bad',))],[predicted])]:
        with pytest.raises(TrainingDataError):
            allele_observation_counts(rows, sets, ['a','b'])
    assert allele_observation_counts([row],[predicted],['a','b'])['n'] == 1


def test_merge_and_empty_ledgers_do_not_require_dummy_model_tensors():
    empty = empty_allele_counts()
    assert finalize_gene(empty, True)['n'] == 0
    assert merge_allele_counts(empty, empty) == empty
    with pytest.raises(TrainingDataError):
        merge_allele_counts(empty, {})
    with pytest.raises(TrainingDataError):
        merge_allele_counts(None, empty)


@pytest.mark.parametrize('field,value', [('n',-1), ('n',True), ('n',1.5), ('nt_top1',1),
                                       ('errors',1), ('true_in_predicted_set',1), ('class_support',1),
                                       ('class_top1',1), ('class_error_support',1), ('errors_within_class',1)])
def test_invalid_counter_ledgers_cannot_produce_rates(field,value):
    counts = empty_allele_counts()
    counts[field] = value
    with pytest.raises(TrainingDataError):
        finalize_gene(counts, True)


def test_missing_counter_fields_and_vocabulary_width_are_rejected():
    with pytest.raises(TrainingDataError):
        finalize_gene({}, False)
    with pytest.raises(ConfigurationError):
        gene_allele_counts(torch.ones(1,2), torch.ones(1,2), ['a'], None)
    with pytest.raises(ConfigurationError):
        gene_allele_counts(torch.ones(1,1), torch.ones(1,1), ['a'], None, threshold=None)


class Echo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Dropout().eval()
        self.calls = 0
    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        return inputs


def runner_fixture():
    cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, has_d=False, use_orientation=False)
    outputs, targets = {}, {}
    for gene in ('v','j'):
        outputs[f'{gene}_allele'] = torch.tensor([[.8,.2],[.1,.9],[.8,.2]])
        targets[f'{gene}_allele'] = torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        for boundary in ('start','end'):
            outputs[f'{gene}_{boundary}'] = torch.tensor([1.,4.,9.])
            targets[f'{gene}_{boundary}'] = torch.tensor([[0.],[2.],[6.]])
    targets['v_supervise'] = torch.tensor([True,False,True])
    outputs['mutation_rate'] = torch.tensor([.2,.4,.8])
    targets['mutation_rate'] = torch.tensor([[.1],[.2],[.3]])
    outputs['productive'] = torch.tensor([.8,.6,.4])
    targets['productive'] = torch.tensor([[1.],[0.],[0.]])
    targets['mutation_rate_valid'] = torch.tensor([1,0,1])
    reference = SimpleNamespace(gene=lambda _: SimpleNamespace(names=['a','b']))
    def adapter(records):
        rows = list(records)
        return ({key:value[rows] for key,value in outputs.items()}, {key:value[rows] for key,value in targets.items()})
    return cfg, outputs, targets, reference, adapter


@pytest.mark.parametrize('batch_size', [1,2,5])
def test_shared_runner_uses_each_observed_denominator_and_never_broadcasts_scalar_axes(batch_size):
    cfg, _, _, ref, adapter = runner_fixture()
    model = Echo()
    report = evaluate_allele_cohort(model, iter(range(3)), adapter, cfg, ref,
                                    {'V':{'a':0,'b':0}}, batch_size=batch_size)
    assert report['n'] == 3
    assert report['alleles']['v']['n'] == 2
    assert report['alleles']['v']['nt_top1'] == .5
    assert report['alleles']['v']['class_top1'] == 1.
    assert report['segmentation_mae']['v_start'] == 2.
    assert report['mutation_mae'] == pytest.approx(.3)
    assert report['productive_acc'] == pytest.approx(2/3)
    assert report['support']['scalars'] == {'mutation_rate':2,'productive':3}
    assert model.calls == (3+batch_size-1)//batch_size
    assert model.training and not model.child.training
    assert current_error_context() == {}


def test_empty_cohort_reports_zero_records_and_disabled_heads_are_explicitly_absent():
    cfg, _, _, ref, adapter = runner_fixture()
    report = evaluate_allele_cohort(Echo(), [], adapter, cfg, ref)
    assert report['n'] == 0 and report['mutation_mae'] == report['productive_acc'] == 0.
    cfg = replace(cfg, use_meta_heads=False)
    report = evaluate_allele_cohort(Echo(), range(3), adapter, cfg, ref)
    assert report['mutation_mae'] is report['productive_acc'] is None
    assert 'scalars' not in report['support']


@pytest.mark.parametrize('failure', ['config','threshold','maps','reference','size','class_map','productivity_truth','productivity_probability'])
def test_runner_preflight_and_productivity_failures_are_classified(failure):
    cfg, outputs, targets, ref, adapter = runner_fixture()
    maps, threshold = {}, .5
    error = ConfigurationError
    if failure == 'config': cfg = cfg.with_amino_acid_heads()
    elif failure == 'threshold': threshold = None
    elif failure == 'maps': maps = []
    elif failure == 'reference': ref = None
    elif failure == 'size': ref = SimpleNamespace(gene=lambda _: SimpleNamespace(names=['a']))
    elif failure == 'class_map': maps = {'V':[]}
    elif failure == 'productivity_truth': targets['productive'][0] = .5; error = TrainingDataError
    else: outputs['productive'][0] = 2.; error = NumericalError
    model = Echo()
    with pytest.raises(error):
        evaluate_allele_cohort(model, range(3), adapter, cfg, ref, maps, threshold=threshold)
    assert model.training and not model.child.training


def test_cohort_productive_accuracy_counts_only_rows_whose_productivity_is_supervised():
    cfg, _, targets, ref, adapter = runner_fixture()
    targets['productive_supervise'] = torch.tensor([True, True, False])
    report = evaluate_allele_cohort(Echo(), range(3), adapter, cfg, ref)
    assert report['productive_acc'] == pytest.approx(1/2)
    assert report['support']['scalars']['productive'] == 2


@pytest.mark.parametrize('policy,rows', [('masked', [1, 2]), ('negative', [0, 1, 2])])
def test_record_cohorts_leave_unknown_productivity_out_under_the_masked_objective(policy, rows):
    from alignair.core import AlignAIR
    from alignair.data import NucleotideBatchAdapter
    from alignair.reference.reference_set import GeneReference, ReferenceSet
    sequence = 'CAGGTGCAGCTGTGCTGGGGCCAG'
    spans = {'v': (0, 12), 'd': (12, 15), 'j': (15, 24)}
    reference = ReferenceSet({gene.upper(): GeneReference([gene.upper() + '*01'], [sequence[start:end]],
                                                         {gene.upper() + '*01': 0})
                              for gene, (start, end) in spans.items()}, has_d=True)
    record = {'sequence': sequence, 'mutation_rate': .05, 'n_indels': 0, 'n_quality_errors': 0}
    for gene, (start, end) in spans.items():
        record.update({f'{gene}_call': gene.upper() + '*01', f'{gene}_sequence_start': start,
                       f'{gene}_sequence_end': end, f'{gene}_germline_start': 0,
                       f'{gene}_germline_end': end - start, f'{gene}_cigar': f'{end - start}M'})
    records = [{**record, 'productive': value, 'sequence_id': str(i)} for i, value in enumerate((None, True, False))]
    cfg = AlignAIRConfig(1, 1, 1, has_d=True, max_seq_length=128, embed_dim=4, filters=4, block_out=8,
                         use_orientation=False, productive_unknown=policy)
    adapter = NucleotideBatchAdapter(reference, cfg)
    observed = collect_alignment_observations(AlignAIR(cfg), records, lambda chunk: tuple(adapter(chunk)), cfg,
                                              scalar_heads=('productive',))
    assert [row.record_index for row in observed.scalars['productive']] == rows
    assert [row.truth for row in observed.scalars['productive']] == [0., 1., 0.][-len(rows):]


def test_default_collector_retains_no_optional_payload():
    cfg, _, _, _, adapter = runner_fixture()
    observed = collect_alignment_observations(Echo(), range(3), adapter, cfg)
    assert not observed.prediction_sets and not observed.scalars


@pytest.mark.parametrize('heads', ['mutation_rate', None, (1,), ('',), ('x','x')])
def test_invalid_optional_scalar_selection_is_rejected(heads):
    cfg, _, _, _, adapter = runner_fixture()
    with pytest.raises(ConfigurationError):
        collect_alignment_observations(Echo(), [], adapter, cfg, scalar_heads=heads)


def test_thresholded_allele_sets_cannot_be_requested_from_categorical_logits():
    cfg, _, _, _, adapter = runner_fixture()
    with pytest.raises(ConfigurationError, match='multilabel'):
        collect_alignment_observations(Echo(), [], adapter, cfg.with_amino_acid_heads(), prediction_set_threshold=.5)
