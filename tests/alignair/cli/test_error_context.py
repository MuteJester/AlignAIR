"""A command's failure report carries the context the command gives it (the stage, the path), also for an error that
is already classified; the error's own context wins where both name a key. Argument-range errors are configuration
errors in every command."""
import json

import pytest

from alignair.cli._errors import fail
from alignair.cli.main import main
from alignair.errors import InputError, error_reporting

pytestmark = pytest.mark.component


def test_a_classified_error_keeps_the_stage_its_command_reports(capsys):
    error = InputError("allele named twice", context={"path": "own.fasta"})
    with error_reporting(json_output=True):
        code = fail(error, context={"stage": "reference_binding", "path": "given.fasta"})
    payload = json.loads(capsys.readouterr().err)
    assert code == int(error.exit_code) and payload["type"] == "InputError"
    assert payload["context"]["stage"] == "reference_binding" and payload["context"]["path"] == "own.fasta"


def test_a_genotype_range_error_is_a_configuration_error_as_in_predict(tmp_path, capsys):
    table = tmp_path / "rows.tsv"
    table.write_text("sequence_id\tv_call\nr0\tIGHV1-2*02\n")
    assert main(["--error-format", "json", "genotype", str(table), "--out", str(tmp_path / "g.json"),
                 "--explain", "2"]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_CONFIG_INVALID"


@pytest.mark.parametrize("requested, loci, expected", [
    ("igh", ("IGH",), "IGH"), (None, ("TRB",), "TRB"), ("trb", (), "TRB"), (None, (), "IGH")])
def test_the_output_locus_is_the_models_spelling_of_the_requested_one(requested, loci, expected):
    """``--locus`` matches the model's loci whatever its case and is written as the model spells it; without it, the
    model's one locus, else IGH."""
    from alignair.cli.predict import output_locus
    assert output_locus(requested, loci) == expected


def test_a_locus_the_model_lacks_is_refused():
    from alignair.cli.predict import output_locus
    from alignair.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match="not one of this model's loci"):
        output_locus("igk", ("IGH",))


def test_the_predict_help_states_the_genotype_format_and_both_token_variables():
    from alignair.cli.main import build_parser
    predict = next(a for a in build_parser()._actions if a.dest == "command").choices["predict"]
    text = " ".join(predict.format_help().split())
    assert "{V|D|J: [allele names]}" in text and "{gene:" not in text
    assert "$HF_TOKEN" in text and "$HUGGING_FACE_HUB_TOKEN" in text
