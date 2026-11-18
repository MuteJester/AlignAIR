"""Text that described removed or wrong behaviour stays out of the code: the package docstrings load a catalog id of
the AlignAIR 4 form (alignair4-<locus>-human), the genotype rule's docstring quotes no unlabelled measured rates, the
assay's allele table does not claim calibrated probabilities, and the dead --update-goldens test option is gone. The
shipped code carries no experiment history (decision numbers, GenAIRR request codes, lab suite codes, dev-only assay
references), the ``.alignair`` container documents its byte layout, and the public API has docstrings."""
import inspect
import re
from pathlib import Path

import pytest

import alignair
import alignair.aligner
import alignair.core.config
import alignair.genotype.infer
import alignair.train.records.streams
from alignair.core.tokens import TOKEN_DICT
from tests.support.public_api import public_api

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", [alignair, alignair.aligner])
def test_package_docstrings_load_an_alignair4_catalog_id(module):
    assert "alignair-igh-v1" not in module.__doc__
    assert re.search(r'from_pretrained\("alignair4-igh-human"', module.__doc__)


def test_the_genotype_rule_quotes_no_unlabelled_rates():
    assert not re.search(r"\d+\.\d+%", alignair.genotype.infer.__doc__)


def test_the_assay_allele_table_does_not_claim_calibrated_probabilities():
    from alignair_assay.report.diagnostics import _suite_heads
    text = "\n".join(_suite_heads({"heads": {"head.v.allele_top1": {"mean": 0.5, "n": 4, "coverage": 1.0}}}))
    assert "calibrated" not in text and "Alleles (ranked by score)" in text


def test_the_training_records_docstring_claims_no_locus_class_tagging():
    assert "class ID" not in alignair.train.records.streams.training_records.__doc__


def test_the_vocab_size_comment_points_at_the_one_vocabulary_instead_of_restating_it():
    line = next(l for l in Path(alignair.core.config.__file__).read_text().splitlines() if "vocab_size: int" in l)
    assert "core.tokens.TOKEN_DICT" in line and not any(f"{k}={v}" in line for k, v in TOKEN_DICT.items())


def test_the_dead_update_goldens_option_is_gone():
    for path in (ROOT / "tests" / "support").glob("*.py"):
        assert "update-goldens" not in path.read_text() and "UPDATE_GOLDENS" not in path.read_text(), path.name


# What a reader of the shipped code cannot decode: the owner's decisions, the GenAIRR request list, the lab's chips and
# suites, the dev-only assay, the experiment's recipes and its placeholders.
_HISTORY = re.compile(r"owner's|\bdecision \d|\(ask G\d|GenAIRR G\d|chip-v1b|alignair_assay|\bassay\b|historical|"
                      r"1\.0\.0 recipe|recipe v\d|PENDING-V4-NUMBERS|today's|dev suites|\bpilot\b|parasail|"
                      r"\(S\d{1,2}[,:]|inference-only")


def test_the_shipped_code_carries_no_experiment_history():
    sources = [path for path in sorted((ROOT / "src" / "alignair").rglob("*.py*")) if path.suffix in (".py", ".pyx")]
    assert len(sources) > 100
    found = [f"{path.relative_to(ROOT)}:{n}: {line.strip()}" for path in sources
             for n, line in enumerate(path.read_text().splitlines(), 1) if _HISTORY.search(line)]
    assert not found, "\n".join(found)


def test_the_container_docstring_documents_the_byte_layout():
    from alignair.model_file import container
    for term in ("ALGNAIR", "u64", "sections", "offset", "compressed_length", "payload_length", "codec",
                 "compressed_sha256", "payload_sha256", "format"):
        assert term in container.__doc__, term


def test_the_model_file_docstrings_name_the_reference_section_and_the_publish_artifact():
    import alignair.model_file as mf
    import alignair.model_file.io as mf_io
    for doc in (mf.__doc__, mf_io.__doc__):
        assert "reference_json" in doc and "include_trusted_pickle=False" in doc


def _own_docstring(value) -> str:
    """``value``'s own docstring: not a base class's, and not the signature a dataclass writes for a class without."""
    doc = (vars(value).get("__doc__") if inspect.isclass(value) else value.__doc__) or ""
    return "" if inspect.isclass(value) and doc.startswith(f"{value.__name__}(") else doc.strip()


def test_the_public_api_has_docstrings():
    missing = sorted({name for name, value in public_api() if not _own_docstring(value)})
    assert not missing, missing


# Words of the model's internals that a repertoire researcher reading a message or --help need not know.
_INTERNAL_WORDS = re.compile(r"\b(neural|chip|trunk|heads|bank|forward pass|embeddings?|curriculum|logits?)\b",
                             re.IGNORECASE)


def test_user_facing_messages_use_plain_words(tmp_path):
    """The refusals a user meets name what to do in their own terms: a reference without the model's D, and an
    AlignAIR 3 model file."""
    import warnings

    import GenAIRR.data as gd

    from alignair import Aligner, ReferenceSet
    from alignair.errors import AlignAIRError
    from alignair.model_file import load_model
    from tests.alignair.model_file.conftest import write_v3_header_file
    from tests.alignair.predict.golden_rows import TEST_CHIP
    messages = []
    with pytest.raises(AlignAIRError) as caught, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Aligner.from_pretrained(str(TEST_CHIP), device="cpu", reference=ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB))
    messages.append(f"{caught.value.message} {caught.value.hint}")
    with pytest.raises(AlignAIRError) as caught:
        load_model(write_v3_header_file(tmp_path))
    messages.append(f"{caught.value.message} {caught.value.hint}")
    assert [m for m in messages if _INTERNAL_WORDS.search(m)] == []


def _help_texts(parser, path=("alignair",)):
    import argparse
    yield " ".join(path), parser.description or ""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for choice in action._choices_actions:
                yield " ".join((*path, choice.dest)), choice.help or ""
            for name, child in action.choices.items():
                yield from _help_texts(child, (*path, name))
        elif action.help:
            yield f"{' '.join(path)} {'/'.join(action.option_strings) or action.dest}", action.help


def test_cli_help_uses_plain_words():
    from alignair.cli.main import build_parser
    found = [f"{where}: {text}" for where, text in _help_texts(build_parser()) if _INTERNAL_WORDS.search(text)]
    assert found == []
