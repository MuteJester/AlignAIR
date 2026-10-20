import gzip
import hashlib
import json

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import genairr_provenance  # noqa: E402
from alignair_assay.suites.donors import (  # noqa: E402
    donors_from_json,
    donors_summary,
    donors_to_json,
    sample_donors,
)
from alignair_assay.suites.engine import derive_seed  # noqa: E402
from alignair_assay.testing import assert_oracle_perfect  # noqa: E402
from alignair_assay.truth.io import read_truth_header, read_truth_jsonl  # noqa: E402
from alignair_assay.workflow import build as build_module  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

REFERENCE = "HUMAN_IGH_OGRDB"


def _build(root, suite="S0", **overrides):
    layout = RunLayout(root)
    options = dict(reference_name=REFERENCE, split="dev", seed=11, scale=1.0, workers=1) | overrides
    return layout, build_suite(suite, layout, **options)


def _content(path):
    return gzip.decompress(path.read_bytes())


def test_build_is_deterministic(tmp_path):
    a, _ = _build(tmp_path / "a", workers=1)
    b, _ = _build(tmp_path / "b", workers=2)
    for path in (RunLayout.cases, RunLayout.truth):
        assert _content(path(a, "S0")) == _content(path(b, "S0"))
    assert a.reads("S0").read_bytes() == b.reads("S0").read_bytes()
    c, _ = _build(tmp_path / "c", seed=12)
    assert _content(c.cases("S0")) != _content(a.cases("S0"))


def test_build_writes_layout(tmp_path, reference):
    layout, build = _build(tmp_path)
    for path in (layout.cases("S0"), layout.truth("S0"), layout.reads("S0"), layout.build_json("S0")):
        assert path.is_file(), path
    assert not layout.donors("S0").exists()  # S0 has no donors

    header = read_truth_header(layout.truth("S0"))
    assert header == {"kind": "alignair_assay.truth", "version": "2", "reference": REFERENCE,
                      "reference_digest": reference.digest(), **genairr_provenance(),
                      "suite": "S0", "split": "dev", "seed": 11, "scale": 1.0, "n_cases": 50}

    cases = read_cases(layout.cases("S0"))
    truths = read_truth_jsonl(layout.truth("S0"))
    assert [t.case_id for t in truths] == [c["case_id"] for c in cases]
    assert all(t.suite == "S0" for t in truths)
    lines = layout.reads("S0").read_text().splitlines()
    assert lines[0::2] == [f">{c['case_id']}" for c in cases] and lines[1::2] == [c["sequence"] for c in cases]

    assert json.loads(layout.build_json("S0").read_text()) == build
    assert (build["suite"], build["split"], build["seed"], build["scale"]) == ("S0", "dev", 11, 1.0)
    assert build["reference"] == REFERENCE and build["reference_digest"] == reference.digest()
    assert build["n_cases"] == 50 and build["seconds"] > 0
    assert {name: s["requested"] for name, s in build["strata"].items()} == {"plain": 30, "with_d": 20}
    assert {name: s["cases"] for name, s in build["strata"].items()} == {"plain": 30, "with_d": 20}
    canonical = json.dumps({"suite": "S0", "split": "dev", "reference_digest": reference.digest(), "seed": 11,
                            "scale": 1.0}, sort_keys=True, separators=(",", ":"))
    assert build["config_hash"] == hashlib.sha256(canonical.encode()).hexdigest()

    summary = build["truth"]
    assert summary["cases"] == 50 and summary["excluded"] == {}
    assert summary["warnings"] == {t.case_id: list(t.warnings) for t in truths if t.warnings}
    assert summary["unscorable"] == {g: sum(g in t.unscorable for t in truths) for g in "vdj"}
    assert summary["simulator_disagreements"] == {
        g: sum(t.segments[g].alleles.disagrees_with_simulator for t in truths if g in t.segments) for g in "vdj"}


def test_config_hash_follows_split_and_scale(tmp_path):
    _, dev = _build(tmp_path / "dev", scale=0.2)
    _, test = _build(tmp_path / "test", scale=0.2, split="test")
    _, larger = _build(tmp_path / "larger", scale=0.4)
    assert len({dev["config_hash"], test["config_hash"], larger["config_hash"]}) == 3


def test_failed_rebuild_leaves_no_build_record(tmp_path, monkeypatch):
    layout, _ = _build(tmp_path, scale=0.2)
    assert layout.build_json("S0").is_file()

    def fail(*args, **kwargs):
        raise RuntimeError("generation failed")

    # build.json marks a complete build, so a rebuild that fails must not leave the old one behind.
    monkeypatch.setattr(build_module, "generate_suite", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        _build(tmp_path, scale=0.4)
    assert not layout.build_json("S0").exists()


def test_toy_suite_oracle_perfect(tmp_path, reference):
    layout, _ = _build(tmp_path)
    assert_oracle_perfect(read_truth_jsonl(layout.truth("S0")), reference)


def test_build_writes_donors(tmp_path, reference):
    layout, build = _build(tmp_path, suite="S0_COHORT")
    donors = sample_donors(reference, 3, derive_seed(11, "dev", "S0_COHORT", "donors", "S0"))
    data = json.loads(layout.donors("S0_COHORT").read_text())
    assert data == donors_to_json(donors) and donors_from_json(data) == donors
    assert build["donors_summary"] == donors_summary(donors, reference)
    assert json.loads(layout.build_json("S0_COHORT").read_text()) == build
    assert build["n_cases"] == 12 and {s["cases"] for s in build["strata"].values()} == {10, 2}

    truths = read_truth_jsonl(layout.truth("S0_COHORT"))
    assert {t.groups["subject"] for t in truths} == {"donor00", "donor01", "donor02"}
    assert_oracle_perfect(truths, reference)

    _, plain = _build(tmp_path)
    assert "donors_summary" not in plain and not layout.donors("S0").exists()
