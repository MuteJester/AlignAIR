import hashlib
import json

import pytest

from alignair.aa.guard import disjoint
from alignair.aa.product.applicability import canonical

pytestmark = pytest.mark.component


@pytest.fixture
def inputs(tmp_path):
    old = {"id": "old", "aa": "AAAA", "lineage_key": "old-lineage", "group": "igsf",
           "category": "igsf_vset", "expected_state": "unknown_ood", "donor": "old-donor", "study": "old-study"}
    splits = disjoint._replay_splits([old])
    digest = canonical.digest([canonical.canonical_record(old, splits[(old["group"], old["lineage_key"])])])
    paths = {key: tmp_path / name for key, name in (
        ("fresh", "fresh.jsonl"), ("cleaned", "cleaned.jsonl"),
        ("fingerprint", "fingerprint.json"), ("gold", "gold.tsv"))}
    paths["cleaned"].write_text(json.dumps(old) + "\n")
    paths["fingerprint"].write_text(json.dumps({"record_digest": {"global": digest}}))
    paths["gold"].write_text("name\texact_vh_id\ntherapeutic\t" + hashlib.sha256(b"CCCC").hexdigest() + "\n")
    paths["fresh"].write_text(json.dumps({"id": "new", "aa": "DDDD", "lineage_key": "new-lineage"}) + "\n")
    return paths


def _args(paths):
    return [str(paths["fresh"]), "--cleaned", str(paths["cleaned"]),
            "--fingerprint", str(paths["fingerprint"]), "--phase3-gold", str(paths["gold"]),
            "--error-format", "json"]


def test_clean_inputs_reproduce_digest_and_return_success(inputs, capsys):
    assert disjoint.main(_args(inputs)) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["clean"] is True
    assert all(value == 0 for value in result["overlap_counts"].values())
    assert result["denylist_sizes"] == dict(sequence_hashes=1, lineages=1, donors=1, studies=1)
    assert not output.err


@pytest.mark.parametrize("field,value,overlap", [
    ("aa", "AAAA", "gate3_seqhash"),
    ("lineage_key", "old-lineage", "gate3_lineage"),
    ("donor", "old-donor", "donor"),
    ("study", "old-study", "study"),
    ("lineage_key", " ThEraPeUtIc ", "phase3_inn"),
    ("aa", "CCCC", "phase3_vh"),
])
def test_every_overlap_dimension_returns_failure(inputs, capsys, field, value, overlap):
    fresh = json.loads(inputs["fresh"].read_text())
    fresh[field] = value
    inputs["fresh"].write_text("\n" + json.dumps(fresh) + "\n\n")
    assert disjoint.main(_args(inputs)) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["clean"] is False
    assert result["overlap_counts"][overlap] == 1
    assert sum(result["overlap_counts"].values()) == 1


def test_wrong_fingerprint_fails_before_reading_fresh_records(inputs, capsys):
    inputs["fingerprint"].write_text('{"record_digest":{"global":"wrong"}}')
    inputs["fresh"].unlink()
    assert disjoint.main(_args(inputs)) != 0
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["code"] == "ALIGNAIR_COHORT_DIGEST_MISMATCH"


@pytest.mark.parametrize("line", ['{"bad"', '[]', '{"id":"new","aa":7}', '{"id":"new","aa":"DDDD","study":[]}'])
def test_bad_fresh_records_report_line_number_without_success_output(inputs, capsys, line):
    inputs["fresh"].write_text("\n" + line + "\n")
    assert disjoint.main(_args(inputs)) != 0
    output = capsys.readouterr()
    assert not output.out
    diagnostic = json.loads(output.err)
    assert diagnostic["context"]["line"] == 2
    assert "traceback" not in diagnostic


def test_missing_labels_cannot_silently_disable_therapeutic_exclusions(inputs, capsys):
    inputs["gold"].unlink()
    assert disjoint.main(_args(inputs)) != 0
    output = capsys.readouterr()
    assert not output.out
    assert "labels are missing" in json.loads(output.err)["message"]
