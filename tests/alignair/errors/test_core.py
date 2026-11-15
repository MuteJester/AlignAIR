import asyncio
import io
import json
import logging
import pickle
from pathlib import Path

import pytest

from alignair.errors import (AlignAIRError, InputError, MissingDependencyError, ModelSectionError,
                             RegistryError, current_error_context, error_context, error_reporting,
                             format_error, log_error, report_error)

pytestmark = pytest.mark.component


def test_builtin_compatibility_and_public_base():
    from alignair import AlignAIRError as public_base
    assert public_base is AlignAIRError
    assert isinstance(InputError("bad input"), (AlignAIRError, ValueError))
    assert isinstance(MissingDependencyError("missing"), ImportError)
    assert isinstance(ModelSectionError("missing section"), KeyError)
    assert isinstance(RegistryError("unavailable"), RuntimeError)
    assert isinstance(RegistryError("unavailable"), ValueError)
    assert str(InputError("bad input")) == "bad input"


def test_context_snapshot_precedence_and_cleanup():
    supplied = {"nested": {"value": 1}}
    with error_context(run_id="outer", stage="read"):
        with error_context(stage="predict"):
            error = InputError("bad", context={**supplied, "stage": "validate"}, hint="Check input.")
            supplied["nested"]["value"] = 2
        assert current_error_context() == {"run_id": "outer", "stage": "read"}
    assert current_error_context() == {}
    assert error.context == {"run_id": "outer", "stage": "validate", "nested": {"value": 1}}
    error.context["nested"]["value"] = 99
    assert error.context["nested"]["value"] == 1
    with pytest.raises(RuntimeError):
        with error_context(stage="fail"):
            raise RuntimeError("fail")
    assert current_error_context() == {}


def test_async_operations_do_not_share_context():
    async def task(run_id):
        with error_context(run_id=run_id):
            await asyncio.sleep(0)
            return InputError("bad").context

    async def run():
        return await asyncio.gather(task("a"), task("b"))

    assert asyncio.run(run()) == [{"run_id": "a"}, {"run_id": "b"}]
    assert current_error_context() == {}


def test_diagnostics_are_bounded_redacted_and_strict_json():
    class Opaque:
        def __repr__(self):
            raise AssertionError("diagnostics must not inspect arbitrary objects")

    cycle = []
    cycle.append(cycle)
    error = InputError("Invalid annotations.", context={
        "hf_token": "private-token", "sequence": "ACGT", "nested": {"password": "private-password"},
        "source": "https://user:password@example.com/model?token=private-token#secret",
        "path": Path("model.alignair"), "loss": float("nan"), "opaque": Opaque(),
        "cycle": cycle, "many": list(range(1000)), "long": "x" * 10000,
    })
    text = error.to_json()
    for secret in ("private-token", "private-password", "ACGT", "user:password", "#secret"):
        assert secret not in text
    payload = json.loads(text, parse_constant=lambda value: pytest.fail(value))
    assert payload["schema_version"] == 1
    assert payload["code"] == "ALIGNAIR_INPUT_INVALID"
    assert payload["context"]["loss"] == "nan"
    assert payload["context"]["opaque"] == "<Opaque>"
    assert payload["context"]["cycle"] == ["[cycle]"]
    assert len(payload["context"]["many"]) == 51
    assert len(payload["context"]["long"]) < 1100


def test_causes_are_preserved_but_only_exposed_in_debug_reports():
    try:
        try:
            raise OSError("upstream-sensitive-detail")
        except OSError as exc:
            raise RegistryError("Registry unavailable.", context={"model_id": "igh"},
                                hint="Check connectivity.") from exc
    except RegistryError as error:
        assert isinstance(error.__cause__, OSError)
        normal = io.StringIO()
        assert report_error(error, stream=normal) == 1
        assert "upstream-sensitive-detail" not in normal.getvalue()
        debug = io.StringIO()
        report_error(error, stream=debug, debug=True, json_output=True)
        payload = json.loads(debug.getvalue())
        assert "upstream-sensitive-detail" in payload["traceback"]
        assert payload["context"]["model_id"] == "igh"


def test_reporting_options_are_scoped_and_stderr_is_used(capsys):
    error = InputError("bad input", hint="Check the sequence file.")
    with error_reporting(json_output=True):
        report_error(error)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["message"] == "bad input"
    report_error(error)
    assert capsys.readouterr().err.startswith("[ALIGNAIR_INPUT_INVALID]")
    assert "Hint: Check" in format_error(error)


def test_logging_is_opt_in_and_retains_structured_diagnostic(caplog):
    logger = logging.getLogger("alignair.test.errors")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        error = InputError("invalid", context={"gene": "D"})
        assert not caplog.records
        log_error(logger, error)
    assert len(caplog.records) == 1
    assert caplog.records[0].alignair_error == error.to_dict()
    assert caplog.records[0].exc_info is None


@pytest.mark.parametrize("kind", [InputError, RegistryError, MissingDependencyError])
def test_pickling_retains_worker_diagnostics(kind):
    error = kind("problem", context={"gene": "D"}, hint="Inspect input.", retryable=True)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is kind
    assert restored.args == error.args
    assert restored.to_dict() == error.to_dict()


def test_an_error_received_from_a_worker_takes_the_receivers_scope_beneath_its_own():
    from alignair.errors.base import rescoped
    received = pickle.loads(pickle.dumps(InputError("problem", context={"stage": "worker", "seed": 3})))
    with error_context(stage="consumer", step=7):
        assert rescoped(received) is received
    assert received.context == {"stage": "worker", "seed": 3, "step": 7}
    assert rescoped(received).context == received.context        # outside any scope: unchanged


def test_legacy_constructor_can_be_pickled():
    from alignair.predict.airr.builder import AirrAssemblyError
    error = AirrAssemblyError("read-1", KeyError("d"))
    restored = pickle.loads(pickle.dumps(error))
    assert restored.ident == "read-1"
    assert isinstance(restored.original, KeyError)
    assert restored.to_dict() == error.to_dict()


def test_an_io_failure_is_an_os_error_for_library_callers():
    import pickle

    from alignair.errors import OperationIOError
    error = OperationIOError("cannot write", context={"path": "out.tsv"})
    assert isinstance(error, OSError) and str(error) == "cannot write" and error.code == "ALIGNAIR_IO_FAILED"
    again = pickle.loads(pickle.dumps(error))
    assert type(again) is OperationIOError and again.context == {"path": "out.tsv"}


def test_a_padding_failure_is_a_data_contract_error_not_a_training_one():
    """``pad_vectors`` also pads the reads of a prediction, so its failure is not reported as a training error."""
    import torch

    from alignair.data.batching import pad_vectors
    from alignair.errors import DataContractError, TrainingDataError
    with pytest.raises(DataContractError) as caught:
        pad_vectors([torch.zeros(5, dtype=torch.long)], width=3)
    assert not isinstance(caught.value, TrainingDataError) and caught.value.code == "ALIGNAIR_DATA_CONTRACT"
