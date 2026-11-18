"""``Aligner.predict``'s shape: the sequences, then by name ``ids``, ``batch_size``, ``threshold``, ``cap`` and ``airr``
and nothing else (the model's window, D presence and loci are the model's, not a call's); ``ids`` name the records and
the result; a bad option is refused before the model runs; an Aligner built without a device predicts on the one
``"auto"`` picks, as every constructor does."""
import inspect
import json

import pytest
import torch

from alignair import Aligner
from alignair.aligner import resolve_device
from alignair.errors import ConfigurationError, InputError
from tests.alignair.predict.golden_rows import GOLDEN_READS, TEST_CHIP

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def aligner():
    return Aligner.from_pretrained(str(TEST_CHIP), device="cpu")


@pytest.fixture(scope="module")
def reads():
    return json.loads(GOLDEN_READS.read_text())["reads"][:5]


def _named(method) -> list:
    return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()][1:]


def test_predict_takes_its_options_by_name_and_no_others():
    keyword = inspect.Parameter.KEYWORD_ONLY
    options = [("ids", keyword, None), ("batch_size", keyword, "auto"), ("threshold", keyword, None),
               ("cap", keyword, None), ("airr", keyword, True)]
    first = ("sequences", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty)
    assert _named(Aligner.predict) == [first, *options]
    assert _named(Aligner.predict_iter) == [first, *options[:2], ("chunk_size", keyword, 20000), *options[2:]]
    assert _named(Aligner.predict_raw) == [first, ("batch_size", keyword, "auto")]


def test_ids_name_the_records_and_the_result(aligner, reads, tmp_path):
    ids = [f"read{i}" for i in range(len(reads))]
    named = aligner.predict(reads, ids=ids, batch_size=8)
    plain = aligner.predict(reads, batch_size=8)
    assert named.ids == ids and plain.ids is None
    assert [record["sequence_id"] for record in named] == ids
    assert [{k: v for k, v in r.items() if k != "sequence_id"} for r in named] == plain.records
    assert "sequence_id" not in plain.records[0]
    path = named.write_airr(str(tmp_path / "named.tsv"))
    assert [line.split("\t")[0] for line in open(path).read().splitlines()[1:]] == ids


@pytest.mark.parametrize("ids", [["a"], ["a", "b", "c", "d", "e", "f"], "abcde", ["a", "b", 3, "d", "e"]],
                         ids=["fewer", "more", "a string", "not strings"])
def test_ids_that_do_not_name_each_sequence_are_refused(aligner, reads, ids):
    with pytest.raises(InputError, match="ids"):
        aligner.predict(reads, ids=ids, batch_size=8)


def test_predict_iter_names_its_records_across_chunks(aligner, reads):
    ids = [f"read{i}" for i in range(len(reads))]
    streamed = list(aligner.predict_iter(iter(reads), ids=iter(ids), batch_size=8, chunk_size=2))
    assert [record["sequence_id"] for record in streamed] == ids
    # a read's record is the one its chunk gives (the reads it is batched with can move its last float bits)
    assert streamed == [record for i in range(0, len(reads), 2)
                        for record in aligner.predict(reads[i:i + 2], ids=ids[i:i + 2], batch_size=8)]
    with pytest.raises(InputError, match="ids"):
        list(aligner.predict_iter(reads, ids=ids[:-1], batch_size=8, chunk_size=2))


def test_threshold_and_cap_set_the_called_alleles(aligner, reads):
    single = aligner.predict(reads, batch_size=8, cap=1)
    assert all(len(record[f"{gene}_calls"]) == 1 for record in single for gene in "vj")
    every = aligner.predict(reads, batch_size=8, threshold=0.0, cap=5)
    assert all(len(record["v_calls"]) == 5 for record in every)


@pytest.mark.parametrize("option, value", [
    ("threshold", float("nan")), ("threshold", "0.5"), ("threshold", True),
    ("cap", 0), ("cap", -1), ("cap", 2.5), ("cap", True),
    ("batch_size", 0), ("batch_size", -4), ("batch_size", 2.5), ("batch_size", "64"),
])
def test_a_bad_option_is_refused_before_the_model_runs(aligner, reads, option, value, monkeypatch):
    ran = []
    monkeypatch.setattr(aligner.model, "forward", lambda *a, **k: ran.append(1))
    with pytest.raises(ConfigurationError) as caught:
        aligner.predict(reads, **{option: value})
    assert caught.value.context["field"] == option and ran == []


def test_a_bad_raw_batch_size_is_refused(aligner, reads):
    with pytest.raises(ConfigurationError) as caught:
        aligner.predict_raw(reads, batch_size=0)
    assert caught.value.context["field"] == "batch_size"


def test_an_aligner_built_without_a_device_predicts_on_the_one_auto_picks():
    assert inspect.signature(Aligner.__init__).parameters["device"].default == "auto"
    model = torch.nn.Linear(2, 2)
    aligner = Aligner(model, reference=None)
    assert aligner.device == resolve_device("auto") and aligner.model is model
    assert next(model.parameters()).device.type == torch.device(aligner.device).type


@pytest.mark.parametrize("chunk_size", [0, -1, 2.5, True])
def test_a_bad_chunk_size_is_refused(aligner, reads, chunk_size):
    with pytest.raises(ConfigurationError) as caught:
        list(aligner.predict_iter(reads, chunk_size=chunk_size))
    assert caught.value.context["field"] == "chunk_size"


def test_batch_size_none_means_auto_as_every_option_s_none_means_its_default(aligner, reads):
    """As in AlignAIR 3, and as a wrapper that forwards an optional ``batch_size`` (the CLI's ``--batch-size``) does."""
    assert aligner.predict(reads, batch_size=None).records == aligner.predict(reads).records
    assert aligner.predict_raw(reads, batch_size=None).keys() == aligner.predict_raw(reads).keys()
    assert [r["v_call"] for r in aligner.predict_iter(reads, batch_size=None)] == [
        r["v_call"] for r in aligner.predict(reads).records]


def test_a_batch_size_string_other_than_auto_is_refused_naming_auto(aligner, reads):
    with pytest.raises(ConfigurationError, match='positive integer or "auto"') as caught:
        aligner.predict(reads, batch_size="AUTO")
    assert caught.value.context["field"] == "batch_size"


def test_predict_iter_with_ids_refuses_a_read_that_is_not_a_string_as_the_input_s_error(aligner, reads):
    import numpy as np
    with pytest.raises(InputError):
        list(aligner.predict_iter([reads[0], np.array(list(reads[1]))], ids=["a", "b"]))
