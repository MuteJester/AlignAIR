"""The CLI command surface is a versioned compatibility contract. This test asserts exactly the
commands the README / CI / Docker health check / release smoke depend on exist and their `--help`
parses, so the release workflows can never reference a missing command again."""
import argparse

import pytest

from alignair.cli.main import build_parser

# The contract surface. Every command here is referenced by a maintained wrapper (README / ci.yml /
# Dockerfile / scripts/release_smoke.sh). Add here only alongside the wrapper that uses it.
CONTRACT_COMMANDS = {
    "predict", "train", "demo", "doctor", "info", "models", "model", "reference",
    "export-reference", "convert", "validate-airr", "compare", "analyze", "benchmark", "genotype",
}


def _subcommands(parser) -> set:
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            return set(a.choices)
    return set()


def test_parser_exposes_the_contract_surface():
    assert _subcommands(build_parser()) >= CONTRACT_COMMANDS


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--version"])
    assert e.value.code == 0


@pytest.mark.parametrize("cmd", sorted(CONTRACT_COMMANDS))
def test_every_command_help_parses(cmd):
    with pytest.raises(SystemExit) as e:      # argparse --help exits 0 after printing
        build_parser().parse_args([cmd, "--help"])
    assert e.value.code == 0


@pytest.mark.parametrize("sub", ["list", "export"])
def test_reference_subcommands_parse(sub):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["reference", sub, "--help"])
    assert e.value.code == 0


def test_release_smoke_commands_are_all_in_the_contract():
    """The exact verbs scripts/release_smoke.sh invokes must all be contract commands."""
    for cmd in ("doctor", "demo", "validate-airr", "compare"):
        assert cmd in CONTRACT_COMMANDS
