"""Public prediction validates batches before forwarding and shares normalized text across stages."""
from types import SimpleNamespace

import pytest
import torch

from alignair import Aligner
from alignair.api import predict_sequences
from alignair.errors import InputError

pytestmark = pytest.mark.component


class ObservedModel(torch.nn.Module):
    def __init__(self, orientation=0):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(max_seq_length=4, has_d=False)
        self.orientation = orientation
        self.tokens = []

    def forward(self, inputs):
        tokens = inputs['tokenized_sequence']
        self.tokens.append(tokens.tolist())
        count = len(tokens)
        return {'v_allele': torch.ones(count, 1), 'j_allele': torch.ones(count, 1),
                'v_start': torch.zeros(count), 'v_end': torch.full((count,), 2.),
                'j_start': torch.full((count,), 2.), 'j_end': torch.full((count,), 4.),
                'productive': torch.ones(count), 'mutation_rate': torch.zeros(count),
                'indel_count': torch.zeros(count), 'orientation': torch.full((count,), self.orientation)}


def reference():
    return SimpleNamespace(gene=lambda gene: SimpleNamespace(names=(f'{gene}1',)))


@pytest.mark.parametrize('entry', ['functional', 'object', 'stream'])
@pytest.mark.parametrize('value', ['ACGT', b'ACGT', {'sequence': 'ACGT'}, None, ['AC', None], ['AC', 123], ['AC', {'sequence': 'AC'}]])
def test_public_entries_reject_malformed_batches_without_running_model(entry, value):
    model = ObservedModel()
    aligner = Aligner(model, reference())
    with pytest.raises(InputError):
        if entry == 'functional':
            predict_sequences(model, aligner.reference, value, airr=False)
        elif entry == 'object':
            aligner.predict(value, airr=False)
        else:
            list(aligner.predict_iter(value, airr=False))
    assert model.tokens == []


@pytest.mark.parametrize('orientation', [0, 1, 2, 3])
def test_normalized_read_reaches_tokens_alignment_and_record_coordinates(monkeypatch, orientation):
    from alignair.predict import pipeline
    model = ObservedModel(orientation)
    alignment_reads = []
    def align(sequences, segments, calls, *args, **kwargs):
        alignment_reads.extend(sequences)
        for gene in ('v', 'j'):
            assert all(0 <= start <= end <= len(sequence)
                       for start, end, sequence in zip(segments.start[gene], segments.end[gene], sequences))
        return {gene: [None] * len(sequences) for gene in ('v', 'j')}
    monkeypatch.setattr(pipeline, 'align_germline', align)
    result = Aligner(model, reference()).predict(iter(['  ac?pT  ']), airr=False)
    assert model.tokens == [[[1, 4, 5, 5]]]
    assert len(result) == 1
    record = result.records[0]
    assert record['input_sequence'] == 'ACNN' and record['length_cropped'] is True
    assert record['sequence'] == pipeline._canonicalize('ACNN', orientation)
    assert alignment_reads == [record['sequence']]
    assert model.training


def test_eager_batch_preflight_stops_before_model_even_with_valid_earlier_rows():
    model = ObservedModel()
    with pytest.raises(InputError) as caught:
        Aligner(model, reference()).predict((value for value in ('AC', '', 'GT')), airr=False)
    assert model.tokens == [] and caught.value.context['record_index'] == 1


def test_genotype_repertoire_probabilities_records_and_lengths_share_prepared_reads(monkeypatch):
    from alignair.genotype.aggregate import align_repertoire
    from alignair.predict import pipeline
    monkeypatch.setattr(pipeline, 'align_germline', lambda sequences, *args, **kwargs:
                        {gene: [None] * len(sequences) for gene in ('v', 'j')})
    model = ObservedModel()
    result = align_repertoire(model, reference(), iter([' ac?pT ', 'g']), device='cpu')
    assert result.sequences == ['ACNN', 'G'] and result.read_lengths.tolist() == [4, 1]
    assert [record['input_sequence'] for record in result.records] == result.sequences
    assert result.records[0]['length_cropped'] is True
    assert model.tokens == [[[1, 4, 5, 5], [3, 0, 0, 0]], [[1, 4, 5, 5], [3, 0, 0, 0]]]
    assert result.allele_probs['v'].shape == (2, 1)


def test_temperature_calibration_uses_shared_preparation_for_generator_records():
    from alignair.calibration import calibrate_allele_temperatures
    model = ObservedModel()
    temperatures, report = calibrate_allele_temperatures(
        model, reference(), iter([{'sequence': ' ac?pT ', 'v_call': 'V1'}]), genes=('v', 'j'))
    assert temperatures == report == {}  # too few labelled examples to fit a temperature
    assert model.tokens == [[[1, 4, 5, 5]]]
