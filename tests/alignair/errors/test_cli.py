import json
import importlib
from types import SimpleNamespace

import pytest

cli = importlib.import_module("alignair.cli.main")
from alignair.errors import InputError

pytestmark = pytest.mark.component


def _command(monkeypatch, exception):
    def run(args):
        raise exception
    def register(sub):
        sub.add_parser("fail").set_defaults(func=run)
    monkeypatch.setattr(cli, "_COMMANDS", (SimpleNamespace(register=register),))


def test_cli_expected_errors_have_context_and_go_to_stderr(monkeypatch, capsys):
    # Create the exception inside the command to capture command context.
    def run(args):
        raise InputError("Invalid record.", context={"record_id": "r1"}, hint="Inspect the record.")
    def register(sub):
        sub.add_parser("fail").set_defaults(func=run)
    monkeypatch.setattr(cli, "_COMMANDS", (SimpleNamespace(register=register),))
    assert cli.main(["--error-format", "json", "fail"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["context"] == {"command": "fail", "record_id": "r1"}
    assert "traceback" not in payload


@pytest.mark.parametrize("debug", [False, True])
def test_unexpected_defects_have_separate_status_and_preserved_traceback(monkeypatch, capsys, debug):
    _command(monkeypatch, KeyError("unexpected-detail"))
    args = (["--debug"] if debug else []) + ["--error-format", "json", "fail"]
    assert cli.main(args) == 70
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["code"] == "ALIGNAIR_INTERNAL_ERROR"
    assert payload["context"]["exception_type"] == "KeyError"
    assert ("traceback" in payload) == debug
    if debug:
        assert "unexpected-detail" in payload["traceback"]
    else:
        assert "unexpected-detail" not in captured.err


def test_interrupt_and_io_failures_are_classified(monkeypatch, capsys):
    _command(monkeypatch, KeyboardInterrupt())
    assert cli.main(["fail"]) == 130
    assert "ALIGNAIR_INTERRUPTED" in capsys.readouterr().err
    _command(monkeypatch, FileNotFoundError(2, "missing", "reads.fasta"))
    assert cli.main(["--error-format", "json", "fail"]) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "ALIGNAIR_IO_FAILED"
    assert payload["context"]["path"] == "reads.fasta"


def test_argparse_retains_usage_exit_status():
    with pytest.raises(SystemExit) as caught:
        cli.main(["--not-an-option"])
    assert caught.value.code == 2


def test_reporting_flags_work_after_commands_and_do_not_reset_parent_options(monkeypatch, capsys):
    _command(monkeypatch, KeyError("details"))
    assert cli.main(["--debug", "fail", "--error-format", "json"]) == 70
    assert "details" in json.loads(capsys.readouterr().err)["traceback"]
    assert cli.main(["--error-format", "json", "fail"]) == 70
    assert "traceback" not in json.loads(capsys.readouterr().err)


def test_nested_reporting_options_are_available():
    args = cli.build_parser().parse_args(["--debug", "models", "path", "igh", "--error-format", "json"])
    assert args.debug is True
    assert args.error_format == "json"


def test_real_command_uses_json_reporting_for_handled_validation_failure(tmp_path, capsys):
    path = tmp_path / "invalid.tsv"
    path.write_text("sequence_id\tsequence\nr1\tACGT\n")
    assert cli.main(["validate-airr", str(path), "--error-format", "json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["code"] == "ALIGNAIR_INPUT_INVALID"
    assert payload["context"]["missing_columns"] == ["v_call", "j_call"]


def test_system_exit_is_not_swallowed(monkeypatch):
    _command(monkeypatch, SystemExit(9))
    with pytest.raises(SystemExit) as caught:
        cli.main(["fail"])
    assert caught.value.code == 9
