"""Real reference/model/adapter combinations agree with the shared task metric support."""
from itertools import islice

import pytest
import torch

from alignair.core import prepare_model
from alignair.data.adapters import CategoricalBatchAdapter, NucleotideBatchAdapter, ProteinAlleleBatchAdapter
from alignair.data.sampling import ProteinSampleBuilder
from alignair.data.tokenizer import AAProteinTokenizer, CenterPaddedTokenizer
from alignair.evaluate import collect_alignment_observations, evaluate_allele_cohort, evaluate_categorical_cohort
from alignair.reference import build_protein_reference
from alignair.tasks import build_tasks
from alignair.train import evaluation_mode
from alignair.train.gym import CurriculumStreamPolicy, training_records

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def native_cohort():
    reference = build_protein_reference('HUMAN_IGH_OGRDB')
    builder = ProteinSampleBuilder(reference.class_maps, AAProteinTokenizer(), 192,
                                   reference.source.gene('D').index)
    selected = []
    stream = training_records(('HUMAN_IGH_OGRDB',), policy=CurriculumStreamPolicy(mode='amplicon'),
                              seed=21, translated_only=True)
    for record in islice(stream, 500):
        sample = builder(record).sample
        if sample is not None and len(record['sequence']) < 576:
            selected.append((record,sample))
            if len(selected) == 3:
                break
    assert len(selected) == 3
    return reference, selected


@pytest.mark.parametrize('input_space,label_space', [('nucleotide','alleles'), ('protein','alleles'),
                                                   ('protein','protein_classes'), ('nucleotide','protein_classes')])
def test_real_alphabet_and_head_combinations_match_task_metrics(native_cohort, input_space, label_space):
    reference, pairs = native_cohort
    setup = prepare_model(reference if label_space == 'protein_classes' else reference.source,
                          input_space=input_space, label_space=label_space,
                          max_seq_length=576 if input_space == 'nucleotide' else 192,
                          embed_dim=4, filters=4, block_out=8)
    cfg = setup.config
    model = setup.build_model()
    if label_space == 'alleles':
        records = [record for record, _ in pairs]
        adapter = (NucleotideBatchAdapter(reference.source, cfg, augment_orientation=False)
                   if input_space == 'nucleotide' else ProteinAlleleBatchAdapter(reference.source, cfg))
    else:
        records = []
        for record, sample in pairs:
            sample = dict(sample)
            if input_space == 'nucleotide':
                sample['tokens'] = CenterPaddedTokenizer().encode(record['sequence'])
                for gene in ('v','d','j'):
                    if sample.get(f'{gene}_valid', 1):
                        for boundary in ('start','end'):
                            sample[f'{gene}_{boundary}'] = record[f'{gene}_sequence_{boundary}']
            records.append(sample)
        adapter = CategoricalBatchAdapter.from_config(cfg)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    observed = collect_alignment_observations(model, iter(records), adapter, cfg, batch_size=2)
    with evaluation_mode(model):
        inputs, targets = adapter(records)
        metrics = build_tasks(cfg).metric_report(model(inputs), targets)
    assert observed.record_count == 3
    for gene, rows in observed.classification.items():
        name = f'{gene}_class_acc' if cfg.classification_mode == 'categorical' else f'{gene}_allele_top1'
        assert len(rows) == metrics.counts.get(name, 0)
        if rows:
            assert sum(row.correct for row in rows)/len(rows) == pytest.approx(metrics.means()[name])
        if cfg.classification_mode == 'categorical':
            for boundary in ('start','end'):
                name = f'{gene}_{boundary}_mae'
                boundaries = observed.boundaries[f'{gene}_{boundary}']
                assert len(boundaries) == metrics.counts.get(name,0)
                if boundaries:
                    assert sum(row.absolute_error for row in boundaries)/len(boundaries) == pytest.approx(metrics.means()[name])
    assert model.training
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())
    if label_space == 'alleles':
        equivalence = {gene: mapping.allele_class for gene,mapping in reference.class_maps.items()}
        report = evaluate_allele_cohort(model, iter(records), adapter, cfg, reference.source,
                                        equivalence, batch_size=2)
        assert report['n'] == 3
        assert report['mutation_mae'] == pytest.approx(metrics.means()['mutation_mae'])
        assert report['productive_acc'] == pytest.approx(metrics.means()['productive_acc'])
        for gene, rows in observed.classification.items():
            assert report['alleles'][gene]['n'] == len(rows)
            if rows:
                assert report['alleles'][gene]['nt_top1'] == pytest.approx(sum(row.correct for row in rows)/len(rows))
        assert model.training
    else:
        report = evaluate_categorical_cohort(model, iter(records), adapter, cfg, batch_size=2)
        assert report['n'] == 3 and report['support'] == observed.support
        assert report['stratified'] == {}
        assert list(report['classification']) == [spec.name for spec in cfg.gene_specs]
        for gene, rows in observed.classification.items():
            if rows:
                assert report['classification'][gene]['micro_acc'] == pytest.approx(sum(row.correct for row in rows)/len(rows))
