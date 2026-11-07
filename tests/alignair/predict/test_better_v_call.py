"""A bound reference's allele that matches more of a read's V bases than the model's call becomes the call
(``predict.ties.better_v_alleles``, V realigned to it by ``predict.germline.realign_v``): on a golden read, an allele
made from the read's called allele by giving it the read's base at one mismatched V position, bound with the model's
reference, is the read's V call and its row is aligned to it, even when the model's V scores favour the base allele
(the two alleles' scores swapped after the forward pass)."""
import json
import re

import pytest

from tests.alignair.predict.golden_rows import FIXTURE, TEST_CHIP

pytestmark = pytest.mark.integration


def _forward(row):
    return row["sequence"] if row.get("orientation_id", 0) == 0 else None


def _variant(row, reference):
    """The called allele with the read's base at the first V position the call's alignment mismatches, or None."""
    read, allele = _forward(row), row["v_call"].split(",")[0]
    if read is None or not allele:
        return None
    germline = reference.gene("V").sequences[reference.gene("V").index[allele]].upper()
    r, g = row["v_sequence_start"], row["v_germline_start"]
    for count, op in re.findall(r"(\d+)([MID])", row["v_cigar"]):
        count = int(count)
        if op == "M":
            for k in range(count):
                if read[r + k] in "ACGT" and read[r + k] != germline[g + k]:
                    return germline[:g + k] + read[r + k] + germline[g + k + 1:]
            r, g = r + count, g + count
        elif op == "I":
            r += count
        else:
            g += count
    return None


def test_an_allele_the_read_matches_better_than_the_call_becomes_the_call(monkeypatch):
    import alignair.predict.pipeline as pipeline
    from alignair.api import load_model, predict_sequences
    from alignair.reference.reference_set import ReferenceSet
    model, reference = load_model(str(TEST_CHIP), device="cpu")
    reads = json.loads(FIXTURE.read_text())["reads"]
    rows = predict_sequences(model, reference, reads, device="cpu")
    i, novel = next((i, v) for i, row in enumerate(rows) if (v := _variant(row, reference)) is not None)
    base = rows[i]["v_call"].split(",")[0]
    NOVEL = base.split("*")[0] + "*99"
    bound = reference.extend(ReferenceSet.from_genotype({"V": {NOVEL: novel}, "J": {}}))
    model, bound = load_model(str(TEST_CHIP), reference=bound, device="cpu")
    index = bound.gene("V").index
    clean = pipeline.clean

    def favouring_the_base(raw, genes):
        preds = clean(raw, genes)
        v = preds.allele["v"]
        v[:, [index[NOVEL], index[base]]] = v[:, [index[base], index[NOVEL]]]
        if v[0, index[NOVEL]] >= v[0, index[base]]:
            v[:, index[NOVEL]] = v[:, index[base]] / 2
        return preds

    monkeypatch.setattr(pipeline, "clean", favouring_the_base)
    row = predict_sequences(model, bound, [reads[i]], device="cpu")[0]
    assert row["v_calls"][0] == base                    # the model's scores still favour the base allele
    assert row["v_call"].split(",")[0] == NOVEL
    assert row["v_germline_end"] <= len(bound.gene("V").sequences[index[NOVEL]])
