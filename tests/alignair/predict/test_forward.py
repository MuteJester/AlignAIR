"""One inference primitive: detached outputs, checked shapes, lazy batches and mode restoration."""
import numpy as np
import pytest
import torch

from alignair.core import evaluation_mode
from alignair.errors import ConfigurationError, InferenceError, InputError, NumericalError, current_error_context
from alignair.predict import PredictConfig, collect_prediction_outputs, forward_batch, prediction_batches
from alignair.predict.forward import PredictionBatch, output_numpy, run_model

pytestmark = pytest.mark.component


class Echo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Dropout().eval()
        self.calls = 0
    def forward(self, inputs):
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        return inputs


def test_selected_outputs_are_detached_and_mixed_modes_restore_before_return():
    model = Echo()
    source = torch.ones(2,3,requires_grad=True)
    result = forward_batch(model,{'scores':source,'unused':torch.full((2,),float('nan'))},
                           expected_rows=2,output_keys=('scores',))
    assert set(result) == {'scores'} and not result['scores'].requires_grad
    assert model.training and not model.child.training
    assert current_error_context() == {}


@pytest.mark.parametrize('outputs', [None,{}, {'x':None}, {'x':torch.tensor(1.)},
                                    {'x':torch.ones(2)}, {'x':torch.ones(1,dtype=torch.complex64)},
                                    {1:torch.ones(1)}, {'':torch.ones(1)}])
def test_bad_outputs_raise_inference_errors_and_restore_modes(outputs):
    model = Echo()
    model.forward = lambda _: outputs
    with pytest.raises(InferenceError):
        forward_batch(model,{},expected_rows=1)
    assert model.training and not model.child.training
    assert current_error_context() == {}


def test_nonfinite_requested_output_has_field_context():
    with pytest.raises(NumericalError) as caught:
        forward_batch(Echo(),{'bad':torch.tensor([float('inf')])},expected_rows=1)
    assert caught.value.context == {'stage':'model_inference','field':'bad'}


@pytest.mark.parametrize('keys', [(), 'x', ('x','x'), (1,), ('',), 1])
def test_invalid_output_selection_fails_before_forward(keys):
    model = Echo()
    with pytest.raises(ConfigurationError):
        forward_batch(model,{},expected_rows=1,output_keys=keys)
    assert model.calls == 0


def test_invalid_assembly_and_missing_keys_are_classified():
    for model, inputs in [(None,{}),(Echo(),None)]:
        with pytest.raises(ConfigurationError):
            forward_batch(model,inputs,expected_rows=1)
    with pytest.raises(InferenceError):
        forward_batch(Echo(),{'other':torch.ones(1)},expected_rows=1,output_keys=('missing',))


@pytest.mark.parametrize('count', [True,False,0,-1,1.5,None])
def test_invalid_row_counts_are_configuration_errors(count):
    with pytest.raises(ConfigurationError):
        forward_batch(Echo(),{},expected_rows=count)


def test_batches_are_lazy_include_tail_and_end_all_execution_scopes_before_yield():
    model, calls = Echo(), []
    def source():
        calls.append('opened')
        yield from range(5)
    def adapter(rows):
        calls.append(tuple(rows))
        return {'values':torch.tensor(rows)}
    batches = prediction_batches(model,source(),adapter,batch_size=2)
    assert calls == [] and model.calls == 0
    first = next(batches)
    assert first.record_offset == 0 and first.size == 2
    assert model.training and not model.child.training and torch.is_grad_enabled()
    assert current_error_context() == {}
    remaining = list(batches)
    assert [(batch.record_offset,batch.size) for batch in remaining] == [(2,2),(4,1)]
    assert calls == ['opened',(0,1),(2,3),(4,)]


def test_empty_batches_do_not_adapt_or_forward_and_invalid_setup_is_eager():
    model = Echo()
    assert list(prediction_batches(model,[],lambda _:pytest.fail('adapted'),batch_size=2)) == []
    for value in (0,True):
        with pytest.raises(ConfigurationError):
            prediction_batches(model,[],lambda _: {},batch_size=value)
    for model,adapter in [(None,lambda _: {}),(Echo(),None)]:
        with pytest.raises(ConfigurationError):
            prediction_batches(model,[],adapter,batch_size=2)


def test_injected_forward_failure_restores_modes():
    model = Echo()
    def fail(_): raise RuntimeError('injected')
    model.forward = fail
    with pytest.raises(RuntimeError,match='injected'):
        forward_batch(model,{},expected_rows=1)
    assert model.training and not model.child.training


def test_numpy_snapshots_are_independent_and_bfloat16_is_representable():
    source = torch.tensor([1.])
    copied = output_numpy(source)
    source.fill_(9)
    assert copied.tolist() == [1.]
    assert output_numpy(torch.ones(2,dtype=torch.bfloat16)).dtype == np.float32


def test_nucleotide_forward_uses_shared_normalization_unknowns_and_cropping():
    model = Echo()
    batches = run_model(model,['acgtn?','a?p','nn'],PredictConfig(max_seq_length=4,batch_size=2))
    # the window is 3 bases: the model's width less the terminal end slot
    assert batches[0]['tokenized_sequence'].tolist() == [[1,4,3,0],[1,5,5,0]]
    assert batches[1]['tokenized_sequence'].tolist() == [[5,5,0,0]]
    assert model.training


def test_historical_evaluation_context_is_the_shared_core_runtime_owner():
    from alignair.train.engine import evaluation_mode as old
    assert old is evaluation_mode


def test_collection_snapshots_reused_storage_and_keeps_tail_and_first_key_order():
    buffer = torch.zeros(2, 3)
    def batches():
        yield PredictionBatch(0, 2, {'scores': buffer, 'positions': torch.tensor([4, 5])})
        buffer.fill_(7)
        yield PredictionBatch(2, 1, {'positions': torch.tensor([6]), 'scores': buffer[:1]})
    result = collect_prediction_outputs(batches())
    assert list(result) == ['scores', 'positions']
    assert result['scores'].tolist() == [[0, 0, 0], [0, 0, 0], [7, 7, 7]]
    assert result['positions'].tolist() == [4, 5, 6]


@pytest.mark.parametrize('outputs', [{}, {'other': torch.ones(1, 3)},
                                   {'scores': torch.ones(1, 3), 'extra': torch.ones(1)}])
def test_collection_rejects_changed_heads_with_batch_context(outputs):
    batches = [PredictionBatch(0, 2, {'scores': torch.ones(2, 3)}), PredictionBatch(2, 1, outputs)]
    with pytest.raises(InferenceError, match='keys') as caught:
        collect_prediction_outputs(batches)
    assert caught.value.context['record_offset'] == 2
    assert caught.value.context['batch_size'] == 1
    assert caught.value.context['expected_keys'] == ['scores']
    assert current_error_context() == {}


@pytest.mark.parametrize('later', [torch.ones(1, 4), torch.ones(1, 3, 1),
                                 torch.ones(1, 3, dtype=torch.float64)])
def test_collection_rejects_changed_width_rank_or_dtype_and_restores_runtime(later):
    model = Echo()
    def adapter(rows):
        return {'scores': torch.ones(2, 3) if len(rows) == 2 else later}
    with pytest.raises(InferenceError, match='shape or dtype') as caught:
        collect_prediction_outputs(prediction_batches(model, range(3), adapter, batch_size=2))
    assert caught.value.context['record_offset'] == 2 and caught.value.context['field'] == 'scores'
    assert model.training and not model.child.training and torch.is_grad_enabled()
    assert current_error_context() == {}


@pytest.mark.parametrize('transformed', [None, np.array(1.), np.ones((1, 3))])
def test_collection_checks_transform_rows_and_array_contract(transformed):
    with pytest.raises(InferenceError, match='one row per input') as caught:
        collect_prediction_outputs([PredictionBatch(0, 2, {'scores': torch.ones(2, 3)})],
                                   transform=lambda _: transformed)
    assert caught.value.context['expected_rows'] == 2 and caught.value.context['field'] == 'scores'


def test_collection_snapshots_transform_buffers_before_advancing():
    buffer = np.zeros((1, 2))
    def batches():
        yield PredictionBatch(0, 1, {'scores': torch.ones(1, 2)})
        buffer.fill(9)
        yield PredictionBatch(1, 1, {'scores': torch.ones(1, 2)})
    result = collect_prediction_outputs(batches(), transform=lambda _: buffer)
    assert result['scores'].tolist() == [[0, 0], [9, 9]]


def test_empty_collection_has_explicit_input_error():
    with pytest.raises(InputError, match='nonempty'):
        collect_prediction_outputs([])


class Oriented(torch.nn.Module):
    """Records each call's (training, grad enabled, orientation given) and returns its tokens and orientation: the
    given one, else 3 for every read."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs):
        tokens = inputs["tokenized_sequence"]
        self.calls.append((self.training, torch.is_grad_enabled(), "orientation" in inputs))
        return {"tokenized_sequence": tokens,
                "orientation": inputs.get("orientation", torch.full((tokens.shape[0],), 3))}


def test_run_model_teacher_forces_the_given_orientations():
    model = Oriented()
    batches = run_model(model, ["ACGT", "AC", "A"], PredictConfig(max_seq_length=8, batch_size=2),
                        orientation=[1, 2, 0])
    assert [batch["orientation"].tolist() for batch in batches] == [[1, 2], [0]]
    assert model.calls == [(False, False, True)] * 2           # every id given: one pass per batch


def test_a_none_orientation_is_the_models_own_from_an_unforced_eval_forward():
    model = Oriented().train()
    (batch,) = run_model(model, ["ACGT", "AC"], PredictConfig(max_seq_length=8, batch_size=2), orientation=[1, None])
    assert batch["orientation"].tolist() == [1, 3]
    assert model.calls == [(False, False, False), (False, False, True)] and model.training


@pytest.fixture(scope="module")
def chip_and_reads():
    import json
    from alignair.api import load_model
    from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP
    model, reference = load_model(str(TEST_CHIP), device="cpu")
    return model, reference, json.loads(FIXTURE.read_text())["reads"]


def _same_rows(a: list, i: int, b: list, j: int) -> bool:
    """Whether row i of batches a equals row j of batches b, bit for bit, in every output."""
    def row(batches, k):
        sizes = [len(next(iter(batch.values()))) for batch in batches]
        n = 0
        for batch, size in zip(batches, sizes):
            if k < n + size:
                return {key: value[k - n] for key, value in batch.items()}
            n += size
        raise IndexError(k)
    x, y = row(a, i), row(b, j)
    return x.keys() == y.keys() and all(np.array_equal(x[key], y[key]) for key in x)


def test_a_reads_outputs_are_the_same_whatever_the_reads_of_other_buckets(chip_and_reads):
    """A read predicted with short reads and the same read predicted with long ones (batch size and threads fixed)
    give bit-identical outputs when those reads fall in other width buckets: reads are batched by bucket, so the
    read's width is its own. Eight threads: at two, a width change happens not to move these reads' floats (the
    harness restores the thread count)."""
    from alignair.core.boundaries import width_bucket
    torch.set_num_threads(8)
    model, reference, reads = chip_and_reads
    cfg = PredictConfig.for_model(model, reference, batch_size=8)
    bucket = lambda read: width_bucket(min(len(read), cfg.window) + 1, cfg.max_seq_length)
    short = [read[:60] for read in reads[:3]]
    long_ = [(read + read)[:400] for read in reads[:3]]
    assert not {bucket(r) for r in reads} & {bucket(r) for r in short + long_}
    assert all(_same_rows(run_model(model, [read, *short], cfg), 0, run_model(model, [read, *long_], cfg), 0)
               for read in reads)


def test_outputs_come_back_in_input_order_in_batch_size_chunks():
    """Reads are batched by width bucket, and their outputs come back in input order, in ``batch_size`` chunks, one row
    per read (no read added)."""
    cfg = PredictConfig(max_seq_length=64, batch_size=2)
    reads = ["ACGTACGTAC" * 5, "AC", "ACGTACGTAC" * 4, "A", "ACG"]          # buckets 64, 32, 64, 32, 32
    alone = [run_model(Echo(), [read], cfg)[0]["tokenized_sequence"][0].tolist() for read in reads]
    model = Echo()
    batches = run_model(model, reads, cfg)
    assert [len(batch["tokenized_sequence"]) for batch in batches] == [2, 2, 1]
    assert [row.tolist() for batch in batches for row in batch["tokenized_sequence"]] == alone
    assert model.calls == 3                                                  # bucket 64: 2 reads; bucket 32: 2 + 1


def test_outputs_that_change_keys_between_batches_are_refused():
    class Shifting(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            tokens = inputs["tokenized_sequence"]
            out = {"tokenized_sequence": tokens}
            if self.calls > 1:
                out["extra"] = tokens
            return out
    with pytest.raises(InferenceError, match="changed between batches"):
        run_model(Shifting(), ["ACGT", "AC", "A"], PredictConfig(max_seq_length=8, batch_size=2))


def test_bucket_batches_are_each_buckets_reads_in_input_order_in_batch_size_chunks():
    from alignair.predict.forward import bucket_batches
    reads = ["A" * 40, "A", "A" * 50, "AC", "A" * 45, "ACG", "A" * 63]      # buckets 64, 32, 64, 32, 64, 32, 64
    assert bucket_batches(reads, max_seq_length=576, batch_size=2) == [[0, 2], [4, 6], [1, 3], [5]]


def test_predict_raw_gives_the_arrays_the_rows_come_from(chip_and_reads):
    """``Aligner.predict_raw`` runs the forward ``predict`` runs (``run_model``: reads batched by width bucket): its
    arrays are the ones a read's row comes from, bit for bit, whatever the other reads' lengths."""
    from alignair.aligner import Aligner
    torch.set_num_threads(8)                        # the harness's two threads hide width effects (it restores them)
    model, reference, reads = chip_and_reads
    reads = [read[:length] for read in reads for length in (90, len(read))]
    raw = Aligner(model, reference, device="cpu").predict_raw(reads, batch_size=4)
    batches = run_model(model, reads, PredictConfig.for_model(model, reference, batch_size=4))
    assert raw.keys() == batches[0].keys()
    assert all(np.array_equal(raw[key], np.concatenate([batch[key] for batch in batches])) for key in raw)


def test_a_reads_outputs_are_the_same_whatever_the_length_of_its_bucket_mates(chip_and_reads):
    """A read batched with a shorter read of its width bucket and with a longer one gives bit-identical outputs: in
    evaluation the chip computes over the bucket's width, not the batch's longest read."""
    from alignair.core.boundaries import WIDTH_BUCKET, width_bucket
    torch.set_num_threads(8)                        # the harness's two threads hide width effects (it restores them)
    model, reference, reads = chip_and_reads
    cfg = PredictConfig.for_model(model, reference, batch_size=2)
    checked = 0
    for read in reads:
        n = min(len(read), cfg.window)
        top = width_bucket(n + 1, cfg.max_seq_length) - 1          # the longest read of the bucket
        low = top - WIDTH_BUCKET + 1                                 # the shortest
        if not low < n < top:
            continue
        shorter, longer = (read * 2)[:low], (read * 2)[:top]
        checked += 1
        assert _same_rows(run_model(model, [read, shorter], cfg), 0, run_model(model, [read, longer], cfg), 0)
    assert checked >= 5
