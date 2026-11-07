"""``v_call`` lists the V call, then the alleles the read cannot tell from it (``predict.ties``): on the golden reads and
the tie reads, where no allele matches a read better than the model's top-1 (``test_better_v_call`` covers the reads
where one does), every row's ``v_call`` starts with the selector's top-1 and names distinct V alleles of the reference;
a short amplicon whose V the call's alignment shows only in part lists more than the call."""
import json

import pytest

from tests.alignair.predict.golden_rows import FIXTURE, TIE_READS, golden_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def golden():
    return golden_model()


def test_v_call_lists_the_call_then_its_ties(golden):
    from alignair.api import predict_sequences
    model, reference = golden
    reads = json.loads(FIXTURE.read_text())["reads"] + json.loads(TIE_READS.read_text())["reads"]
    rows = predict_sequences(model, reference, reads, device="cpu")
    names = set(reference.gene("V").names)
    called = [row for row in rows if row["v_call"]]
    assert called
    for row in called:
        listed = row["v_call"].split(",")
        assert listed[0] == row["v_calls"][0]
        assert len(set(listed)) == len(listed) and set(listed) <= names
    assert any("," in row["v_call"] for row in called)
