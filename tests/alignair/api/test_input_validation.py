"""Public prediction validates batches before forwarding and shares normalized text across stages."""
from types import SimpleNamespace

import pytest
import torch

from alignair import Aligner
from alignair.core.orientation import orient
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
                'mutation_rate': torch.zeros(count),
                'indel_count': torch.zeros(count), 'orientation': torch.full((count,), self.orientation)}


def reference():
    return SimpleNamespace(gene=lambda gene: SimpleNamespace(names=(f'{gene}1',)), locus_names=lambda: (),
                           default_locus='IGH')


@pytest.mark.parametrize('entry', ['object', 'stream'])
@pytest.mark.parametrize('value', ['ACGT', b'ACGT', {'sequence': 'ACGT'}, None, ['AC', None], ['AC', 123], ['AC', {'sequence': 'AC'}]])
def test_public_entries_reject_malformed_batches_without_running_model(entry, value):
    model = ObservedModel()
    aligner = Aligner(model, reference())
    with pytest.raises(InputError):
        if entry == 'object':
            aligner.predict(value, airr=False)
        else:
            list(aligner.predict_iter(value, airr=False))
    assert model.tokens == []


@pytest.mark.parametrize('orientation', [0, 1, 2, 3])
def test_normalized_read_reaches_tokens_alignment_and_record_coordinates(monkeypatch, orientation):
    from alignair.predict import stages
    model = ObservedModel(orientation)
    alignment_reads = []
    def align(sequences, segments, calls, *args, **kwargs):
        alignment_reads.extend(sequences)
        for gene in ('v', 'j'):
            assert all(0 <= start <= end <= len(sequence)
                       for start, end, sequence in zip(segments.start[gene], segments.end[gene], sequences))
        return {gene: [None] * len(sequences) for gene in ('v', 'j')}
    monkeypatch.setattr(stages, 'align_germline', align)
    result = Aligner(model, reference()).predict(iter(['  ac?pT  ']), airr=False)
    assert model.tokens == [[[1, 4, 5, 0]]]           # a 3-base window: the width less the terminal end slot
    assert len(result) == 1
    record = result.records[0]
    assert record['input_sequence'] == 'ACN' and record['length_cropped'] is True
    assert record['sequence'] == orient('ACN', orientation)
    assert alignment_reads == [record['sequence']]
    assert model.training


def test_eager_batch_preflight_stops_before_model_even_with_valid_earlier_rows():
    model = ObservedModel()
    with pytest.raises(InputError) as caught:
        Aligner(model, reference()).predict((value for value in ('AC', '', 'GT')), airr=False)
    assert model.tokens == [] and caught.value.context['record_index'] == 1
