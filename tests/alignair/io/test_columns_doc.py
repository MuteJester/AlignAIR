"""Every output column has its meaning in ``io.columns_doc`` (the documentation's column dictionary is generated from
it), and the facts users trip on are stated: identities are 0-1 fractions, T/F columns are T or F, the candidate set
is named once."""
import pytest

from alignair.io.airr import COLUMNS

pytestmark = pytest.mark.component


def test_every_output_column_is_documented():
    from alignair.io.columns_doc import COLUMN_DOCS
    assert list(COLUMN_DOCS) == list(COLUMNS) and len(COLUMNS) == 99
    assert [column for column, doc in COLUMN_DOCS.items() if not doc.meaning.strip()] == []


def test_the_facts_users_trip_on_are_stated():
    from alignair.io.columns_doc import COLUMN_DOCS
    assert {COLUMN_DOCS[f"{g}_identity"].unit for g in "vdj"} == {"fraction 0-1"}
    assert {COLUMN_DOCS[c].kind for c in ("productive", "rev_comp", "segmentation_low_quality")} == {"T/F"}
    assert all("candidate set" in COLUMN_DOCS[f"{g}_call_set"].meaning for g in "vdj")
    assert "_dupN" in COLUMN_DOCS["sequence_id"].meaning and "575" in COLUMN_DOCS["length_cropped"].meaning
    assert all("j_cropped" in COLUMN_DOCS[c].meaning for c in ("airr_assembly_reason", "length_cropped"))


def test_only_the_v_call_lists_tied_alleles():
    # predict/results.py lists the V ties after the call; d_call and j_call name one allele.
    from alignair.io.columns_doc import COLUMN_DOCS
    assert "equally well" in COLUMN_DOCS["v_call"].meaning
    assert [g for g in "dj" if "equally well" in COLUMN_DOCS[f"{g}_call"].meaning] == []
    assert all("one allele" in COLUMN_DOCS[f"{g}_call"].meaning for g in "dj")
