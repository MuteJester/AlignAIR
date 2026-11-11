"""Text that described removed or wrong behaviour stays out of the code: the package docstrings load a catalog id of
the AlignAIR 4 form (alignair4-<locus>-human), the genotype rule's docstring quotes no unlabelled measured rates, the
assay's allele table does not claim calibrated probabilities, and the dead --update-goldens test option is gone."""
import re
from pathlib import Path

import pytest

import alignair
import alignair.aligner
import alignair.core.config
import alignair.genotype.lean
import alignair.train.gym.streams
from alignair.data.tokenizer import TOKEN_DICT

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", [alignair, alignair.aligner])
def test_package_docstrings_load_an_alignair4_catalog_id(module):
    assert "alignair-igh-v1" not in module.__doc__
    assert re.search(r'from_pretrained\("alignair4-igh-human"', module.__doc__)


def test_the_genotype_rule_quotes_no_unlabelled_rates():
    assert not re.search(r"\d+\.\d+%", alignair.genotype.lean.__doc__)


def test_the_assay_allele_table_does_not_claim_calibrated_probabilities():
    from alignair_assay.report.diagnostics import _suite_heads
    text = "\n".join(_suite_heads({"heads": {"head.v.allele_top1": {"mean": 0.5, "n": 4, "coverage": 1.0}}}))
    assert "calibrated" not in text and "Alleles (ranked by score)" in text


def test_the_training_records_docstring_claims_no_locus_class_tagging():
    assert "class ID" not in alignair.train.gym.streams.training_records.__doc__


def test_the_vocab_size_comment_gives_the_tokenizers_order():
    line = next(l for l in Path(alignair.core.config.__file__).read_text().splitlines() if "vocab_size: int" in l)
    order = ", ".join(f"{k}={v}" for k, v in sorted(TOKEN_DICT.items(), key=lambda kv: kv[1]))
    assert order in line


def test_the_dead_update_goldens_option_is_gone():
    for path in (ROOT / "tests" / "support").glob("*.py"):
        assert "update-goldens" not in path.read_text() and "UPDATE_GOLDENS" not in path.read_text(), path.name
