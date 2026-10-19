import importlib.metadata
import re
from pathlib import Path

from alignair_assay.cli import main
from alignair_assay.core.jsonl import read_header, write_jsonl
from alignair_assay.core.provenance import genairr_provenance, sha256_file

GENAIRR_FIELDS = {"genairr_version", "genairr_source", "genairr_git_commit", "genairr_git_dirty"}


def test_genairr_provenance_fields(genairr):
    prov = genairr_provenance()
    assert set(prov) == GENAIRR_FIELDS
    assert prov["genairr_version"] == importlib.metadata.version("GenAIRR")
    assert prov["genairr_source"].endswith("GenAIRR")
    assert prov["genairr_source"] == str(Path(genairr.__file__).resolve().parent)
    commit = prov["genairr_git_commit"]
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)
    assert prov["genairr_git_dirty"] in ((None,) if commit is None else (True, False))


def test_truth_header_records_genairr_commit(tmp_path, load_fixture, reference):
    cases = tmp_path / "cases.jsonl"
    write_jsonl([load_fixture("amplicon_janchor_000000.json")], cases)
    out = tmp_path / "truth.jsonl.gz"
    assert main(["truth", "--bench-cases", str(cases), "--out", str(out)]) == 0
    header = read_header(out)
    assert header["reference_digest"] == reference.digest()
    assert {k: header[k] for k in GENAIRR_FIELDS} == genairr_provenance()


def test_sha256_file(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
