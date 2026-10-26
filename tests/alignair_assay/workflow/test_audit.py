"""``audit``, the readiness gate (``workflow.audit``): every check with fakes, the verdict, the report and the command.

Each check is ``{name, status, n, evidence, reason}``. The stored runs are fabricated records (an ``all`` run's
report, scores and builds; a ``real`` run's summary), or built for real at a tiny scale (S0); the runs a check makes
(``run_all``, ``run_real``, ``evaluate``, ``run_speed``) are fakes or oracle runners. The gated end-to-end audit is not
here: it needs the data, Docker, the model, a GPU and IgBLAST, and is run by hand.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from alignair_assay.cli import main
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.records import write_json, write_jsonl
from alignair_assay.report.audit import audit_markdown
from alignair_assay.workflow import audit
from alignair_assay.workflow.audit import CHECKS, GROUPS, AuditConfig, check, settle, verdict

pytestmark = pytest.mark.genairr_build  # these tests build GenAIRR suites


def _record(checks: list[dict]) -> dict:
    """An audit record as ``run_audit`` writes it, around ``checks``."""
    return {"kind": audit.AUDIT_KIND, "version": audit.AUDIT_VERSION, "verdict": verdict(checks),
            "config": {"sim_run": "/runs/sim", "real_run": "/runs/real", "data_root": "/data", "device": "cuda"},
            "checkpoint": {"label": "base", "path": "/models/m.alignair", "sha256": "ab" * 32},
            "gating": list(audit.GATING),
            "provenance": {"host": "box", "assay": {"git_commit": "c0ffee" * 6 + "abcd", "git_dirty": False},
                           "genairr": {"genairr_version": "2.2.0", "genairr_source_sha256": "12" * 32}},
            "created_utc": "2026-10-21T12:00:00+00:00", "seconds": 600.0, "checks": checks, "groups": GROUPS}


# --- the verdict --------------------------------------------------------------------------------------------------

def test_a_check_has_its_shape_and_passing_on_nothing_fails():
    found = check("null.all", "pass", 250, {"summary": "zero"})
    assert found == {"name": "null.all", "status": "pass", "n": 250, "evidence": {"summary": "zero"},
                     "reason": None}
    with pytest.raises(ValueError, match="unknown check status"):
        check("null.all", "ok", 1, {})
    empty = settle(check("null.all", "pass", 0, {"summary": "nothing"}))
    assert empty["status"] == "fail" and "checked nothing" in empty["reason"]
    assert settle(check("band.speed", "skipped", 0, {}, "skipped (load)"))["status"] == "skipped"


def test_the_audit_passes_only_when_nothing_fails_and_something_passed():
    passed = check("null.all", "pass", 250, {})
    warned = check("self_check.real_truth_health", "warn", 10, {}, "flags")
    skipped = check("band.speed", "skipped", 0, {}, "skipped (load)")
    failed = check("null.real", "fail", 2000, {}, "Δ")
    assert verdict([passed, warned, skipped])["status"] == "pass"
    found = verdict([passed, warned, skipped, failed])
    assert found["status"] == "fail" and found["failed"] == ["null.real"]
    assert found["line"] == "AUDIT FAIL: 4 checks, 1 pass, 1 fail, 1 warn, 1 skipped; failed: null.real"
    nothing = verdict([warned, skipped])
    assert nothing["status"] == "fail" and nothing["line"].endswith("; nothing was checked")
    assert verdict([])["status"] == "fail"
    # a check that can fail the audit, skipped: never a pass, whatever else passed
    gating = check("reproducible.sim_build", "skipped", 0, {}, "skipped (GenAIRR moved)")
    assert "reproducible.sim_build" in audit.GATING and "band.speed" not in audit.GATING
    found = verdict([passed, warned, skipped, gating])
    assert found["status"] == "incomplete" and found["skipped_gating"] == ["reproducible.sim_build"]
    assert found["line"] == ("AUDIT INCOMPLETE: 4 checks, 1 pass, 0 fail, 1 warn, 2 skipped; skipped, and able to "
                             "fail the audit: reproducible.sim_build")
    assert verdict([passed, gating, failed])["status"] == "fail"  # a failure still fails


def test_the_gating_checks_are_every_check_but_the_informational_ones():
    assert set(audit.GATING) == set(CHECKS) - set(audit.INFORMATIONAL)
    assert list(audit.GATING) == [name for name in CHECKS if name in audit.GATING]  # in report order
    assert "self_check.sim_staleness" in audit.INFORMATIONAL  # it warns, never fails


def test_the_report_opens_with_the_verdict_and_lists_every_check_by_group():
    checks = [check("self_check.sim_oracle", "pass", 33119, {"summary": "oracle perfect on 12 of 12 suites"}),
              check("self_check.real_truth_health", "warn", 160733, {"summary": "3 flags", "flags": {"W-97": []}},
                    "a flag warns"),
              check("null.all", "fail", 250, {"summary": "S1/shared: 40 metrics", "problems": ["S1/shared x: Δ +1"]},
                    "not null"),
              check("band.speed", "skipped", 0, {"summary": "load 3.2"}, "skipped (load): the load is high")]
    md = audit_markdown(_record(checks))
    lines = md.splitlines()
    assert lines[0] == "AUDIT FAIL: 4 checks, 1 pass, 1 fail, 1 warn, 1 skipped; failed: null.all"
    # the gates column: whether a check can fail the audit
    assert "| Check | Gates | Status | n | Evidence |" in md
    assert "| self_check.sim_oracle | yes | pass | 33,119 | oracle perfect on 12 of 12 suites |" in md
    assert "| null.all | yes | **fail** | 250 | S1/shared: 40 metrics |" in md
    assert "| band.speed | no | skipped | 0 | load 3.2 |" in md
    positions = [md.index(f"\n## {title}\n") for title in ("Self-checks", "Null comparisons", "IgBLAST bands")]
    assert positions == sorted(positions) and "\n## Reproducibility\n" not in md  # a group without checks
    assert "### null.all: fail (n = 250)" in md and "Reason: not null" in md
    assert '"problems": [' in md  # the evidence, in full
    assert "`/models/m.alignair`" in md and "abababababab" in md and "/runs/sim" in md
    assert "**live** (not a frozen copy" in md  # no freeze record: the audit ran the live GenAIRR
    record = _record(checks)
    record["provenance"]["genairr_freeze"] = {"genairr_git_commit": "3de212ee1932" + "0" * 28, "source": "/g/src"}
    assert "a frozen copy of commit `3de212ee1932` from /g/src" in audit_markdown(record)


def test_the_check_order_covers_every_group():
    assert {name.split(".")[0] for name in CHECKS} == set(GROUPS)
    assert len(CHECKS) == len(set(CHECKS)) == 16


# --- the stored runs' self-checks and IgBLAST bands -----------------------------------------------------------------

pytest.importorskip("GenAIRR")

from alignair_assay.scoring.pipeline import write_scores  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

REFERENCE_NAME = "HUMAN_IGH_OGRDB"
PERFECT = {"v.top1_in_set": 1.0, "j.top1_in_set": 1.0, "v.mutation_count_err": 0.0}


def _sim_run(root: Path, *, suites=("S0",), tools=("igblast", "alignair"), split="test", speed=None) -> RunLayout:
    """A stored ``all`` run's records: its manifest, an oracle scored perfectly on 3 cases per suite and its
    ``report.json`` (with ``speed`` when given)."""
    layout = RunLayout(root)
    config = {"suites": list(suites), "tools": list(tools), "split": split, "reference": REFERENCE_NAME,
              "seed": 7, "scale": 0.2, "igblast_dir": str(root / "igblast")}
    write_json(layout.manifest, {"config": config, "genairr": {}})
    for name in suites:
        layout.scores_dir(name, "shared").mkdir(parents=True)
        write_scores({f"{name}:{k}": dict(PERFECT) for k in range(3)}, layout.tool_scores(name, "shared", "oracle"))
    write_json(layout.report_json, {"suites": [{"name": name, "oracle": {"status": "perfect"}} for name in suites],
                                    "speed": speed})
    return layout


def test_the_stored_oracle_is_checked_from_its_scores(tmp_path):
    sim = _sim_run(tmp_path / "sim", suites=("S0", "S1"))
    found = audit.sim_oracle(sim)
    assert found["status"] == "pass" and found["n"] == 6
    assert found["evidence"]["suites"]["S1"] == {"status": "perfect", "cases": 3, "problems": 0,
                                                 "recorded": "perfect"}
    assert found["evidence"]["split"] == "test"
    write_scores({"S1:0": {**PERFECT, "v.top1_in_set": 0.0}}, sim.tool_scores("S1", "shared", "oracle"))
    found = audit.sim_oracle(sim)
    assert found["status"] == "fail" and "S1" in found["reason"] and "S0" not in found["reason"]
    sim.tool_scores("S0", "shared", "oracle").unlink()  # not scored: unchecked, never perfect
    assert audit.sim_oracle(sim)["evidence"]["suites"]["S0"]["status"] == "not scored"


@pytest.mark.parametrize("recorded", ["imperfect", None])
def test_the_stored_oracle_must_agree_with_what_its_report_recorded(tmp_path, recorded):
    # the scores are perfect, the run's report says otherwise (or nothing): the stored run contradicts itself
    sim = _sim_run(tmp_path / "sim", suites=("S0", "S1"))
    report = json.loads(sim.report_json.read_text())
    report["suites"][1]["oracle"] = {"status": recorded} if recorded else {}
    write_json(sim.report_json, report)
    found = audit.sim_oracle(sim)
    assert found["status"] == "fail" and found["evidence"]["suites"]["S1"]["recorded"] == recorded
    assert "S1" in found["reason"] and "S0" not in found["reason"] and "recorded" in found["reason"]


def test_stale_builds_of_the_stored_run_are_a_warning(tmp_path):
    sim = _sim_run(tmp_path / "sim")
    build_suite("S0", sim, reference_name=REFERENCE_NAME, split="test", seed=7, scale=0.2, workers=1)
    found = audit.sim_staleness(sim)
    assert found["status"] == "pass" and found["n"] == 1 and found["evidence"]["stale_builds"] == {}
    record = json.loads(sim.build_json("S0").read_text())
    record["identity"]["truth_source"] = "0" * 64  # built by other truth code
    write_json(sim.build_json("S0"), record)
    found = audit.sim_staleness(sim)
    assert found["status"] == "warn" and found["evidence"]["stale_builds"] == {"S0": ["truth_source"]}
    assert "informational" in found["reason"] and "truth_source" in found["evidence"]["summary"]


def _igblast_report(layout: RunLayout, mean, n=5000) -> None:
    write_json(layout.tool_report_json("S1", "shared", "igblast"),
               {"tool": "igblast", "overall": {"v.top1_in_set": {"mean": mean, "n": n, "n_applicable": n,
                                                                 "coverage": 1.0}}})


def test_the_stored_igblast_s1_v_top1_is_read_as_consistent_with_the_p2_record_or_warns(tmp_path):
    # informational: the stored run's number was computed by older code; band.sim_v_top1_current is the gate
    assert "band.sim_v_top1" in audit.INFORMATIONAL
    sim = _sim_run(tmp_path / "sim", suites=("S1",))
    for mean, status in ((0.9786, "pass"), (0.969, "pass"), (0.989, "pass"), (0.9685, "warn"), (0.9895, "warn"),
                         (None, "warn")):
        _igblast_report(sim, mean)
        found = audit.sim_v_top1_band(sim)
        assert found["status"] == status, mean
        assert found["n"] == 5000
        assert status == "pass" or "informational" in found["reason"]
    _igblast_report(sim, 0.9786)
    found = audit.sim_v_top1_band(sim)
    assert "97.86%" in found["evidence"]["summary"] and "consistent with the P2 record" in found["evidence"]["summary"]
    other = _sim_run(tmp_path / "other", suites=("S1",), tools=("alignair-local@a", "alignair-local@b"))
    assert audit.sim_v_top1_band(other)["status"] == "warn"


def _donor(status="passed", flagged=(), known=(0.995, 0.999), *, block_status="done", pipeline=True, decoys=True):
    genes = [{"gene": gene, "class": "carried", "reads": 12, "clonotypes": 11, "alleles": {"IGHVF1-G1*01": 12},
              "h0_only": False} for gene in flagged]
    summary = {"real.v.known_top1": {"mean": known[0], "n": 2500}, "real.v.ref_known_top1": {"mean": known[1],
                                                                                             "n": 5000}}
    return {"status": block_status,
            "oracle_self_check": {"status": status, "reads": 2600, "scored": 2500, "ref_reads": 5000,
                                  "pipeline": {"reads": 2600, "agrees": 2600, "passed": pipeline},
                                  "decoys": {"passed": decoys}},
            "truth_health": {"reads": 50000, "genes": genes, "flagged": list(flagged), "flag_clonotypes": 10},
            "tools": {"shared": {"igblast": {"summary": summary}},
                      "genotype": {"igblast": {"summary": {"real.v.ref_known_top1": {"mean": 0.98, "n": 5000}}}}}}


def _real_run(root: Path, donors: dict) -> RunLayout:
    layout = RunLayout(root)
    write_json(layout.real_summary_json(), {"config": {"reference": REFERENCE_NAME, "split": "dev"},
                                            "sensitivity": {"genes": []}, "donors": donors})
    return layout


DEV_DONORS = ("SC-23", "SC-11", "W-97", "W-133")


def _dev(**donors) -> dict:
    """Every dev donor's block (``_donor()``), the given ones (by name, ``-`` as ``_``) replaced."""
    return {name: donors.get(name.replace("-", "_"), _donor()) for name in DEV_DONORS}


def test_every_dev_donor_must_pass_its_self_check(tmp_path):
    real = _real_run(tmp_path / "real", {**_dev(), "SC-4": _donor(status="failed")})
    found = audit.real_oracle(real)
    assert found["status"] == "pass" and found["n"] == 10000  # the test donor SC-4 is not read
    assert set(found["evidence"]["donors"]) == set(DEV_DONORS)
    for bad in (_donor(status="unchecked"), _donor(pipeline=False), _donor(decoys=False),
                _donor(block_status="stopped")):
        found = audit.real_oracle(_real_run(tmp_path / "bad", _dev(SC_23=bad)))
        assert found["status"] == "fail" and "SC-23" in found["reason"] and "W-97" not in found["reason"]


def test_every_dev_donor_must_be_in_the_stored_run(tmp_path):
    # the pilot's dev donors, not the ones the stored run happens to hold: a missing one fails every donor check
    partial = {name: block for name, block in _dev().items() if name != "W-133"}
    real = _real_run(tmp_path / "real", partial)
    for check_donors in (audit.real_oracle, audit.real_truth_health, audit.real_known_band):
        found = check_donors(real)
        assert found["status"] == "fail" and "W-133" in found["reason"] and "missing" in found["reason"], found
    assert audit.real_oracle(_real_run(tmp_path / "none", {}))["status"] == "fail"


def test_truth_health_flags_warn_and_never_fail(tmp_path):
    real = _real_run(tmp_path / "real", _dev(W_97=_donor(flagged=("IGHV3-23",))))
    found = audit.real_truth_health(real)
    assert found["status"] == "warn" and found["n"] == 200000
    assert found["evidence"]["flags"] == {"W-97": [{"gene": "IGHV3-23", "reads": 12, "clonotypes": 11,
                                                    "alleles": {"IGHVF1-G1*01": 12}, "h0_only": False}]}
    assert "W-97 IGHV3-23" in found["evidence"]["summary"]
    assert audit.real_truth_health(_real_run(tmp_path / "clean", _dev()))["status"] == "pass"
    # a donor without a truth-health block was never audited: a warning, not a clean pass
    unaudited = _donor()
    del unaudited["truth_health"]
    found = audit.real_truth_health(_real_run(tmp_path / "unaudited", _dev(SC_11=unaudited)))
    assert found["status"] == "warn" and "SC-11" in found["reason"] and found["evidence"]["unaudited"] == ["SC-11"]


def test_igblast_known_answers_on_the_shared_arm_must_reach_99_percent(tmp_path):
    # the genotype arm's ref_known_top1 (0.98 here) is not held to the band: the donor's genotype may not hold it
    found = audit.real_known_band(_real_run(tmp_path / "real", _dev(SC_23=_donor(known=(0.99, 1.0)))))
    assert found["status"] == "pass" and found["n"] == 10000
    assert "W-97 known_top1 99.5% (n=2500), ref_known_top1 99.9% (n=5000)" in found["evidence"]["summary"]
    for known in ((0.989, 1.0), (1.0, 0.9899), (None, 1.0)):
        found = audit.real_known_band(_real_run(tmp_path / "low", _dev(SC_23=_donor(known=known))))
        assert found["status"] == "fail" and "SC-23" in found["reason"] and "W-97" not in found["reason"], known


# --- reproducibility ----------------------------------------------------------------------------------------------

def _config(tmp_path: Path, **overrides) -> AuditConfig:
    """An audit of S0-sized suites (10 cases) with one worker: the ruling's sizes shrunk for a test."""
    checkpoint = tmp_path / "model.alignair"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"weights")
    (tmp_path / "data").mkdir(exist_ok=True)
    fields = dict(sim_run=str(tmp_path / "sim"), real_run=str(tmp_path / "real"), data_root=str(tmp_path / "data"),
                  checkpoint=str(checkpoint), workers=1, threads=1, seed=7, repro_suite="S0", repro_scale=0.2,
                  null_suites=("S0",), null_scale=0.2, diagnose_suites=("S0",),
                  diagnose_variants=("baseline",), speed_scale=0.2)
    return AuditConfig(**(fields | overrides))


def test_two_builds_on_one_genairr_must_be_byte_identical(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    found = audit.sim_reproducible(cfg, RunLayout(tmp_path / "one"), RunLayout(tmp_path / "two"))
    assert found["status"] == "pass" and found["n"] == 10
    assert set(found["evidence"]["files"]) == {"cases", "truth", "reads"}
    assert all(entry["first"] == entry["second"] for entry in found["evidence"]["files"].values())
    assert len(set(found["evidence"]["genairr_digests"])) == 1

    # a generator that is not deterministic: the second build's reads differ
    builds = []

    def drifting(name, layout, **kwargs):
        built = build_suite(name, layout, **kwargs)
        builds.append(name)
        if len(builds) == 2:
            with open(layout.reads(name), "a", encoding="utf-8") as handle:
                handle.write(">extra\nACGT\n")
        return built

    monkeypatch.setattr(audit, "build_suite", drifting)
    found = audit.sim_reproducible(cfg, RunLayout(tmp_path / "three"), RunLayout(tmp_path / "four"))
    assert found["status"] == "fail" and "reads" in found["reason"] and "cases" not in found["reason"]


@pytest.mark.parametrize("lost, where", [(("reads",), "second"), (("cases", "truth", "reads"), "both")])
def test_two_builds_without_their_cases_truth_and_reads_fail(tmp_path, monkeypatch, lost, where):
    # cases, truth and reads are required in both builds (donors stays optional): a build missing one compared less,
    # and two builds missing all three compared nothing
    builds = []

    def losing(name, layout, **kwargs):
        built = build_suite(name, layout, **kwargs)
        builds.append(name)
        if where == "both" or len(builds) == 2:
            for label in lost:
                getattr(layout, label)(name).unlink()
        return built

    monkeypatch.setattr(audit, "build_suite", losing)
    found = audit.sim_reproducible(_config(tmp_path), RunLayout(tmp_path / "one"), RunLayout(tmp_path / "two"))
    assert found["status"] == "fail" and all(label in found["reason"] for label in lost), found["reason"]
    assert "missing" in found["reason"]


def test_a_genairr_that_moved_during_the_builds_skips_the_comparison(tmp_path, monkeypatch):
    digests = iter(["one", "one", "two"])  # edited between the second build and the last look
    monkeypatch.setattr(audit, "genairr_digest", lambda: next(digests))
    found = audit.sim_reproducible(_config(tmp_path), RunLayout(tmp_path / "one"), RunLayout(tmp_path / "two"))
    assert found["status"] == "skipped" and found["reason"].startswith("skipped (GenAIRR moved)")
    assert "files" not in found["evidence"] and found["n"] == 0


# A synthetic real donor (as ``test_real``'s): shared V alleles with IMGT gaps whose codon 104 is TGT, their IUIS
# labels, a donor carrying IGHV1-2*02 on both haplotypes, and reads of IGHV1-2 and IGHV3-11 clones.
GAPS = ((27, 30), (72, 84), (165, 171), (195, 198))
TAIL = "TGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG"
DONOR = "W-97"


def _random(n: int, seed: int, alphabet: str = "ACGT") -> str:
    import random

    return "".join(random.Random(seed).choices(alphabet, k=n))


def _gapped(seed: int, *mutated: int) -> str:
    bases = list(_random(320, seed))
    bases[309:312] = "TGT"
    for column in mutated:
        bases[column] = "A" if bases[column] != "A" else "C"
    for start, end in GAPS:
        bases[start:end] = "." * (end - start)
    return "".join(bases)


def _synthetic():
    """``(reference, iuis, genotype, reads)`` of the synthetic donor."""
    from alignair_assay.core.reference import Reference
    from alignair_assay.real.genotype import IuisMap, RealAllele, RealGenotype
    from alignair_assay.real.truth import cys104_end

    shared = {"IGHVF1-G1*01": (_gapped(1), ("IGHV1-2*02",)), "IGHVF1-G1*02": (_gapped(1, 150), ("IGHV1-2*04",)),
              "IGHVF7-G1*01": (_gapped(7), ("IGHV3-11*01",))}
    reference = Reference(alleles={"v": {name: gapped.replace(".", "") for name, (gapped, _) in shared.items()},
                                   "d": {"IGHD3-10*01": _random(31, 20)}, "j": {"IGHJ4*02": _random(48, 30)}},
                          anchors={}, gapped_v={name: gapped for name, (gapped, _) in shared.items()})
    labels = {**{name: names for name, (_, names) in shared.items()},
              **{name: (name,) for table in ("d", "j") for name in reference.alleles[table]}}
    sequences = reference.sequences()
    by_sequence = {sequences[name]: tuple(sorted(names)) for name, names in labels.items()}
    iuis = IuisMap(labels_by_shared={name: by_sequence[sequence] for name, sequence in sequences.items()},
                   shared_by_sequence={sequence: (name,) for name, sequence in sequences.items()},
                   labels_by_sequence=by_sequence, reference_digest=reference.digest(),
                   source={"sha256": "ogrdb", "release_version": 8})
    v1, v311 = reference.seq("IGHVF1-G1*01"), reference.seq("IGHVF7-G1*01")
    alleles = tuple(RealAllele(allele_id="IGHV1-2*02", gene="IGHV1-2", segment="v", sequence=v1,
                               gapped=shared["IGHVF1-G1*01"][0], haplotype=haplotype, functional=True,
                               shared=("IGHVF1-G1*01",), iuis=("IGHV1-2*02",)) for haplotype in (1, 2))
    genotype = RealGenotype(donor=DONOR, alleles=alleles, captured=frozenset({"IGHV1-2", "IGHV3-11"}),
                            resolved=frozenset({"IGHV1-2", "IGHV3-11"}), uncalled=frozenset(),
                            source={"kind": "test", "sha256": "csv", "reference_digest": reference.digest()})
    end = cys104_end(shared["IGHVF1-G1*01"][0])
    reads = []
    for k in range(6):
        exact = v1[:end] + _random(18, 100 + k, "ACT") + TAIL
        mutated = exact[:10] + ("A" if exact[10] != "A" else "C") + exact[11:]
        reads += [(f"v1_{k}a", exact, "IGHM"), (f"v1_{k}b", mutated, "IGHG")]
    for k in range(2):
        read = v311[:end] + _random(21, 200 + k, "ACT") + TAIL
        reads += [(f"v311_{k}{m}", read, "IGHG") for m in "ab"]
    return reference, iuis, genotype, reads


def _stamp(path: Path) -> None:
    """Rewrite a gzip file with a time in its header, as JSON-lines files were written before their gzip was
    reproducible: the same content, other bytes."""
    import gzip

    data = gzip.decompress(path.read_bytes())
    with open(path, "wb") as handle, gzip.GzipFile(filename=path.name, mode="wb", fileobj=handle,
                                                   mtime=1_792_583_362) as archive:
        archive.write(data)


@pytest.fixture
def stored_real(tmp_path, monkeypatch):
    """A stored ``real`` run of the synthetic donor as ``run_real`` left it: reads, reads.tsv, genotype and fetch
    records, the truth, the self-check (the pipeline oracle's TSV and scores), two runs' predictions, records and
    scores, and the summary; the reference and the IUIS map are loaded through ``real_steps`` (patched). Its truth and
    scores files have a time in their gzip header, as the stored runs written before reproducible gzip do, so the
    scores' inputs name the stamped truth file's digest."""
    import csv

    from alignair_assay.core.provenance import sha256_file
    from alignair_assay.real.genotype import genotype_to_json, merge_identical
    from alignair_assay.real.preprocess import TSV_COLUMNS
    from alignair_assay.workflow import real_score, real_steps
    from alignair_assay.workflow.real import donor_scoring, scoring_inputs, self_check

    reference, iuis, genotype, reads = _synthetic()
    monkeypatch.setattr(real_steps, "load_reference", lambda name: reference)
    monkeypatch.setattr(real_steps, "load_iuis", lambda path, ref: iuis)
    layout = RunLayout(tmp_path / "real")
    layout.real_donor_dir(DONOR).mkdir(parents=True)
    layout.real_reads(DONOR).write_text("".join(f">{name}\n{sequence}\n" for name, sequence, _ in reads))
    with open(layout.real_reads_meta(DONOR), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows({"sequence_id": name, "isotype": isotype, "protocol": "multiplex", "consensus_count": 1,
                          "dupcount": 1, "primer_masked": "[]"} for name, _, isotype in reads)
    write_json(layout.real_genotype_json(DONOR), {"kind": real_steps.GENOTYPE_KIND,
                                                  "genotype": genotype_to_json(genotype)})
    write_json(layout.real_fetch_json(DONOR), {"records": {"genotypes": {"ogrdb": {"path": "ogrdb.json"}}}})
    truths, _, _ = real_steps.step_truth(DONOR, layout, genotype, reference, iuis)
    _stamp(layout.real_truth(DONOR))
    data = donor_scoring(DONOR, layout, truths, genotype, iuis, reference, merge_identical(reference), ["IGHV3-11"])
    common = scoring_inputs(DONOR, layout, iuis, reference)
    oracle = self_check(data, layout, common)
    runs = {"shared": {}, "genotype": {}}
    for tool, arm in (("igblast", "shared"), ("alignair-api", "genotype")):
        predictions = layout.real_predictions(tool, DONOR, arm)
        predictions.parent.mkdir(parents=True)
        calls = ["sequence_id\tsequence\tv_call\td_call\tj_call"]
        for name, sequence, _ in reads:
            call = "IGHVF1-G1*01" if name.startswith("v1_") else "IGHVF7-G1*01"
            if tool == "alignair-api" and name.endswith("b"):
                call = "IGHVF1-G1*02"
            calls.append(f"{name}\t{sequence}\t{call}\tIGHD3-10*01\tIGHJ4*02")
        predictions.write_text("\n".join(calls) + "\n")
        write_json(layout.real_run_json(tool, DONOR, arm), {"provenance": {"predictions_sha256":
                                                                            sha256_file(predictions)}})
        real_score.step_scores(data, tool, arm, layout, {**common, "predictions_sha256": sha256_file(predictions)})
        _stamp(layout.real_scores(DONOR, arm, tool))
        runs[arm][tool] = {"status": "ok"}
    _stamp(layout.real_scores(DONOR, "shared", "oracle"))
    write_json(layout.real_summary_json(), {"config": {"reference": "SYNTHETIC", "split": "dev"},
                                            "sensitivity": {"genes": ["IGHV3-11"]},
                                            "donors": {DONOR: {"status": "done", "oracle_self_check": oracle,
                                                               "runs": runs}}})
    return layout


def _rewrite(path: Path, edit) -> None:
    """Rewrite a JSON-lines record file through ``edit(header, rows)``, which changes them in place."""
    from alignair_assay.core.jsonl import read_header, read_jsonl

    header, rows = read_header(path), read_jsonl(path)
    edit(header, rows)
    write_jsonl(path, rows, header=header)


def _snapshot(root: Path) -> dict[str, str]:
    """Every file under ``root`` by relative path, with its SHA-256."""
    from alignair_assay.core.provenance import sha256_file

    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_real_donors_truth_and_scores_recompute_to_the_stored_ones(tmp_path, stored_real):
    cfg = _config(tmp_path)
    before = _snapshot(stored_real.root)
    found = audit.real_reproducible(cfg, stored_real, RunLayout(tmp_path / "scratch"))
    assert _snapshot(stored_real.root) == before  # the stored run is only read: every file as it was, none added
    assert found["status"] == "pass" and found["n"] == 16, found["reason"]
    files = found["evidence"]["files"]
    assert set(files) == {"truth", "shared/oracle", "shared/igblast", "genotype/alignair-api"}
    # the same content and inputs: the stored scores name the stamped truth file's digest, the recomputed ones the
    # recomputed truth file's, and the two truths are the same content
    assert all(entry["identical"] and not entry["code_moved"] for entry in files.values())
    assert not any(entry["file_identical"] for entry in files.values())  # the stored files carry a gzip time
    assert found["evidence"]["oracle_tsv_identical"] and found["evidence"]["self_check_identical"]
    assert found["evidence"]["runs_compared"] == 2


def test_scores_recomputed_by_moved_code_compare_by_content(tmp_path, stored_real):
    # the stored scores were computed by other scoring code (another digest in the header): the bytes differ, the
    # rows and inputs do not
    def moved(header, rows):
        header.update(code_sha256="0" * 64, fingerprint="1" * 64)

    _rewrite(stored_real.real_scores(DONOR, "genotype", "alignair-api"), moved)
    found = audit.real_reproducible(_config(tmp_path), stored_real, RunLayout(tmp_path / "scratch"))
    assert found["status"] == "pass"
    entry = found["evidence"]["files"]["genotype/alignair-api"]
    assert entry["identical"] and entry["code_moved"] and not entry["file_identical"]
    assert found["evidence"]["code_moved"] == ["genotype/alignair-api"]


@pytest.mark.parametrize("target", ["scores", "input", "truth", "self_check", "oracle_tsv", "missing_scores",
                                    "no_runs"])
def test_a_stored_record_that_does_not_recompute_fails(tmp_path, stored_real, target):
    def flip(header, rows):
        rows[0]["v"] = "unsupported_allele" if rows[0]["v"] != "unsupported_allele" else "supported"

    def other_truth(header, rows):  # scored from another truth file than the stored one: no alias stands for it
        header["inputs"]["truth_sha256"] = "0" * 64

    if target == "scores":
        _rewrite(stored_real.real_scores(DONOR, "shared", "igblast"), flip)
        named = "shared/igblast"
    elif target == "input":
        _rewrite(stored_real.real_scores(DONOR, "genotype", "alignair-api"), other_truth)
        named = "genotype/alignair-api"
    elif target == "truth":
        _rewrite(stored_real.real_truth(DONOR), lambda header, rows: rows.reverse())
        named = "truth"
    elif target == "self_check":
        summary = json.loads(stored_real.real_summary_json().read_text())
        summary["donors"][DONOR]["oracle_self_check"]["scored"] += 1
        write_json(stored_real.real_summary_json(), summary)
        named = "self-check"
    elif target == "oracle_tsv":  # the pipeline oracle's AIRR TSV, which its scores are read from
        path = stored_real.real_predictions("oracle", DONOR, "shared")
        path.write_text(path.read_text().replace("IGHVF1-G1*01", "IGHVF1-G1*02", 1))
        named = "oracle TSV"
    elif target == "missing_scores":  # an ok run whose stored scores are gone is not skipped
        stored_real.real_scores(DONOR, "shared", "igblast").unlink()
        named = "shared/igblast"
    else:  # no ok run: nothing but the truth and the oracle would be compared
        summary = json.loads(stored_real.real_summary_json().read_text())
        summary["donors"][DONOR]["runs"] = {"shared": {"igblast": {"status": "failed"}}}
        write_json(stored_real.real_summary_json(), summary)
        named = "no run's scores"
    found = audit.real_reproducible(_config(tmp_path), stored_real, RunLayout(tmp_path / "scratch"))
    assert found["status"] == "fail" and named in found["reason"], found["reason"]


# --- null comparisons ---------------------------------------------------------------------------------------------

from alignair_assay.runners import VARIANTS  # noqa: E402
from alignair_assay.runners import alignair as alignair_module  # noqa: E402
from alignair_assay.runners.oracle import OracleRunner  # noqa: E402
from alignair_assay.runners.variants import AlignAIRLocalVariant  # noqa: E402
from alignair_assay.stats import compare, read_changes, to_dict  # noqa: E402


class OracleVariant(AlignAIRLocalVariant):
    """alignair-local@<label> with its real fingerprint, whose runs write oracle rows; a label in ``perturbed`` calls
    the first read's V allele wrong."""

    perturbed: set = set()

    def prepare(self, layout, reference):
        self._oracle = OracleRunner(layout)
        return {}

    def run(self, fasta, out_tsv, **kwargs):
        result = self._oracle.run(fasta, out_tsv, **kwargs)
        if self.label in OracleVariant.perturbed:
            lines = Path(out_tsv).read_text().splitlines()
            column = lines[0].split("\t").index("v_call")
            cells = lines[1].split("\t")
            cells[column] = "IGHV1-2*99"
            lines[1] = "\t".join(cells)
            Path(out_tsv).write_text("\n".join(lines) + "\n")
        return dataclasses.replace(result, tool=self.name, provenance={})


@pytest.fixture
def oracle_variants(monkeypatch):
    OracleVariant.perturbed = set()
    monkeypatch.setitem(VARIANTS, "alignair-local", OracleVariant.from_label)
    monkeypatch.setattr(alignair_module, "local_code_identity", lambda: {"code_sha256": "c" * 64,
                                                                         "torch_version": None})
    return OracleVariant


def test_all_with_one_checkpoint_under_two_labels_must_compare_to_zero(tmp_path, oracle_variants):
    null, oracle = audit.all_null(_config(tmp_path), RunLayout(tmp_path / "null_all"))
    assert null["name"] == "null.all" and null["status"] == "pass", null["reason"]
    assert null["n"] == 10 and null["evidence"]["problem_count"] == 0
    shared = null["evidence"]["suites"]["S0/shared"]
    assert shared["metrics"] > 5 and shared["predictions_identical"] and shared["n"] == shared["offered"] == 10
    assert null["evidence"]["tools"] == ["alignair-local@a", "alignair-local@b"]
    assert oracle["name"] == "self_check.sim_oracle_current" and oracle["status"] == "pass" and oracle["n"] == 10


def test_all_with_a_read_called_otherwise_is_not_null(tmp_path, oracle_variants):
    oracle_variants.perturbed = {"b"}
    null, oracle = audit.all_null(_config(tmp_path), RunLayout(tmp_path / "null_all"))
    assert null["status"] == "fail" and "not null" in null["reason"]
    # the wrong V call loses v.top1_in_set on one read, and the germline metrics no longer apply to it on that side
    assert null["evidence"]["problem_count"] > 3
    assert all(problem.startswith("S0/shared v.") for problem in null["evidence"]["problems"])
    assert not null["evidence"]["suites"]["S0/shared"]["predictions_identical"]
    assert oracle["status"] == "pass"  # the oracle is not the perturbed tool


def test_all_is_run_again_when_genairr_moves_during_it(tmp_path, oracle_variants, monkeypatch):
    # GenAIRR edited (or committed) while all ran makes its builds stale before they are scored: all is run again
    # once, and skipped when GenAIRR moves again (never a failure of the comparison)
    runs = []
    original = audit.run_all
    monkeypatch.setattr(audit, "run_all", lambda cfg, layout: runs.append(cfg) or original(cfg, layout))
    looks = iter(["one", "two", "two", "two"])
    monkeypatch.setattr(audit, "genairr_identity", lambda: next(looks))
    null, oracle = audit.all_null(_config(tmp_path), RunLayout(tmp_path / "moved_once"))
    assert len(runs) == 2 and null["status"] == "pass" and oracle["status"] == "pass"
    assert null["evidence"]["genairr_moved"] == 1
    runs.clear()
    looks = iter(["one", "two", "three", "four"])
    null, oracle = audit.all_null(_config(tmp_path), RunLayout(tmp_path / "moving"))
    assert len(runs) == 2
    for found in (null, oracle):
        assert found["status"] == "skipped" and found["reason"].startswith("skipped (GenAIRR moved)")


def _compared(a: dict, b: dict) -> dict:
    """``stats.compare`` and ``stats.read_changes`` of two per-read score sets, as ``compare_models`` stores them."""
    metrics = sorted({metric for scores in a.values() for metric in scores})
    return {"metrics": metrics, "comparison": to_dict(compare(a, b, metrics, lower_is_better=set())),
            "changes": read_changes(a, b, metrics, lower_is_better=set())}


def _scores(n=20, *, wrong=(), uncovered=(), nan=False, metrics=audit.SIM_HEADLINE, nan_metrics=()) -> dict:
    """Per-read scores of ``n`` reads: 1.0 on every metric, 0.0 on junction.nt_exact for ``wrong`` reads, NaN
    (applicable, not covered) on it for ``uncovered`` ones, NaN on every read for ``nan_metrics``, and NaN everywhere
    with ``nan``."""
    import math

    def value(metric, k):
        if nan or metric in nan_metrics or (metric == "junction.nt_exact" and k in uncovered):
            return math.nan
        return 0.0 if metric == "junction.nt_exact" and k in wrong else 1.0

    return {f"S1:{k}": {metric: value(metric, k) for metric in metrics} for k in range(n)}


# The head metrics of a fake diagnosis: one boundary metric per gene and edge, and the orientation.
HEAD_METRICS = (*(f"head.{gene}.{edge}_exact.soft" for gene in "vdj" for edge in ("start", "end")),
                "head.orientation.correct")


def _fake_diagnosis(monkeypatch, *, same=False, wrong=(), uncovered=(), both_uncovered=(), nan=False, scored=20,
                    baseline=True, heads=True, heads_unpaired=(), head_metrics=HEAD_METRICS,
                    baseline_metrics=audit.SIM_HEADLINE, dropped=()):
    """Stand-ins for the builds, the model load and ``evaluate``: the file's key, the in-memory model's key (the same
    with ``same``) and their comparison on 20 reads of each suite but the ``dropped`` ones, of which ``scored`` are
    offered as scored (None: the result records no ``reads.scored``). The in-memory model gets ``wrong`` reads wrong
    and does not cover ``uncovered``; neither covers ``both_uncovered``; ``nan``: no score on either side; no
    end-to-end family without ``baseline``, no head family without ``heads``; ``heads_unpaired``: head metrics
    neither side scores; the families compare ``head_metrics`` and ``baseline_metrics``."""
    import types

    from alignair_assay.diagnostics import replay
    from alignair_assay.workflow import diagnose

    calls = {}
    monkeypatch.setattr(audit, "build_suite", lambda name, layout, **kwargs: calls.setdefault("built", []).append(
        name))
    monkeypatch.setattr(audit, "truth_reference", lambda layout, names: "reference")
    monkeypatch.setattr(replay, "load_model_checked", lambda checkpoint, reference, device: ("model", "set"))

    def evaluate(checkpoint, layout, names, **kwargs):
        calls["evaluate"] = (checkpoint, list(names), kwargs)
        common = tuple(both_uncovered)
        sides = [_scores(nan=nan, metrics=head_metrics, nan_metrics=heads_unpaired) for _ in range(2)]
        ends = (_scores(wrong=wrong, uncovered=(*uncovered, *common), nan=nan, metrics=baseline_metrics),
                _scores(uncovered=common, nan=nan, metrics=baseline_metrics))
        suites = {name: {"cases": 20, "clusters": {"clusters": 0},
                         "heads": _compared(*sides) if heads else None,
                         "baseline": _compared(*ends) if baseline else None}
                  for name in names if name not in dropped}
        against = {"baseline": "k1" if same else "k2", "source": "in-memory model", "same_model": same,
                   "suites": suites}
        reads = {} if scored is None else {"scored": scored}
        return types.SimpleNamespace(model_key="k1", against=against,
                                     suites={name: {"reads": dict(reads)} for name in names})

    monkeypatch.setattr(diagnose, "evaluate", evaluate)
    return calls


def test_diagnose_against_itself_in_memory_must_compare_to_zero(tmp_path, monkeypatch):
    calls = _fake_diagnosis(monkeypatch)
    cfg = _config(tmp_path, diagnose_suites=("S1", "S3"))
    found = audit.diagnose_null(cfg, RunLayout(tmp_path / "null_diagnose"))
    assert found["status"] == "pass" and found["n"] == 40, found["reason"]
    assert calls["built"] == ["S1", "S3"]
    checkpoint, names, kwargs = calls["evaluate"]
    assert checkpoint == cfg.checkpoint and names == ["S1", "S3"]
    assert kwargs["against"] == ("model", "set") and kwargs["profile"] is False  # the in-memory form: two keys
    assert kwargs["variants"] == ("baseline",)
    assert found["evidence"]["model_key"] == "k1" and found["evidence"]["baseline"] == "k2"
    assert found["evidence"]["suites"]["S1"]["baseline"] == {"metrics": 5, "paired_metrics": 5, "fixed": 0,
                                                             "broken": 0, "n": 20}
    assert found["evidence"]["suites"]["S1"]["offered"] == 20


def test_diagnose_of_one_model_key_or_with_changed_reads_fails(tmp_path, monkeypatch):
    _fake_diagnosis(monkeypatch, same=True)
    found = audit.diagnose_null(_config(tmp_path), RunLayout(tmp_path / "one"))
    assert found["status"] == "fail" and "one model key" in found["evidence"]["problems"][0]
    _fake_diagnosis(monkeypatch, wrong=(3, 7))  # the baseline gets two reads wrong
    found = audit.diagnose_null(_config(tmp_path), RunLayout(tmp_path / "two"))
    assert found["status"] == "fail"
    problems = found["evidence"]["problems"]
    assert any("junction.nt_exact: Δ +0.1" in problem for problem in problems)
    assert any("2 reads fixed, 0 broken" in problem for problem in problems)
    # a read one side does not cover: every paired Δ is 0 and no read changes, but the two sides' own scores differ
    _fake_diagnosis(monkeypatch, uncovered=(5,))
    found = audit.diagnose_null(_config(tmp_path), RunLayout(tmp_path / "three"))
    assert found["status"] == "fail"
    assert found["evidence"]["problems"] == ["S0 baseline junction.nt_exact: each side's own scores differ",
                                             "S0 baseline junction.nt_exact: pairs 19 of the 19 and 20 reads the two "
                                             "sides score"]


def test_null_comparisons_that_pair_nothing_fail(tmp_path, monkeypatch):
    # every score NaN on both sides: every Δ is None and nothing differs, but nothing was compared
    monkeypatch.setattr(audit, "_EXAMPLES", 100)  # the evidence keeps every problem, not only the first few
    _fake_diagnosis(monkeypatch, nan=True)
    found = audit.diagnose_null(_config(tmp_path), RunLayout(tmp_path / "diagnose"))
    assert found["status"] == "fail" and found["n"] == 0
    problems = found["evidence"]["problems"]
    assert "S0 heads: no metric pairs a read" in problems and "S0 baseline: no metric pairs a read" in problems
    assert "S0 baseline v.top1_in_set: pairs no read" in problems
    _fake_real(monkeypatch, nan=True)
    found = audit.real_null(_config(tmp_path), RunLayout(tmp_path / "real"))
    assert found["status"] == "fail" and found["n"] == 0
    assert f"{DONOR} shared overall: no metric pairs a read" in found["evidence"]["problems"]
    _fake_all(monkeypatch, nan=True)
    null, _ = audit.all_null(_config(tmp_path), RunLayout(tmp_path / "all"))
    assert null["status"] == "fail" and null["n"] == 0
    assert "S0/shared: no metric pairs a read" in null["evidence"]["problems"]


def test_a_null_comparison_rests_on_its_fewest_paired_headline_reads(tmp_path, monkeypatch):
    # junction.nt_exact applies to 19 of the 20 reads (neither side scores read 4): n is 19 per suite
    _fake_diagnosis(monkeypatch, both_uncovered=(4,))
    found = audit.diagnose_null(_config(tmp_path, diagnose_suites=("S1", "S3")), RunLayout(tmp_path / "one"))
    assert found["status"] == "pass" and found["n"] == 38, found["reason"]
    assert found["evidence"]["suites"]["S1"]["baseline"]["n"] == 19
    # more reads offered than any headline metric pairs: some read never reached the comparison
    _fake_diagnosis(monkeypatch, scored=25)
    found = audit.diagnose_null(_config(tmp_path), RunLayout(tmp_path / "two"))
    assert found["status"] == "fail"
    assert "S0 baseline: its headline metrics pair at most 20 of the 25 reads offered" in found["evidence"]["problems"]
    # no end-to-end family, or a head boundary metric pairing nothing (orientation pairs, boundaries do not)
    _fake_diagnosis(monkeypatch, baseline=False)
    found = audit.diagnose_null(_config(tmp_path), RunLayout(tmp_path / "three"))
    assert found["status"] == "fail" and found["n"] == 0
    assert "S0: no end-to-end family was compared (no variant ran)" in found["evidence"]["problems"]
    _fake_diagnosis(monkeypatch, heads_unpaired=("head.v.start_exact.soft",))
    found = audit.diagnose_null(_config(tmp_path), RunLayout(tmp_path / "four"))
    assert found["status"] == "fail"
    assert "S0 heads head.v.start_exact.soft: pairs no read" in found["evidence"]["problems"]


@pytest.mark.parametrize("fault, expected", [
    (dict(baseline_metrics=tuple(metric for metric in audit.SIM_HEADLINE if metric != "d.top1_in_set")),
     "S1 baseline d.top1_in_set: a headline metric, not compared"),
    (dict(heads=False), "S1: no head family was compared"),
    (dict(head_metrics=tuple(metric for metric in HEAD_METRICS if not metric.startswith("head.j.end_"))),
     "S1 heads: no head.j.end_* boundary metric"),
    (dict(dropped=("S3",)), "the comparison holds the suites S1, not the diagnosed S1, S3"),
    (dict(scored=None), "S1: the diagnosis records no scored reads (reads.scored)"),
], ids=["headline_metric", "heads_family", "boundary_metric", "suite", "reads_scored"])
def test_diagnose_without_what_it_must_compare_is_not_null(tmp_path, monkeypatch, fault, expected):
    # whatever is there pairs every read and differs by nothing: what is missing fails on its own
    _fake_diagnosis(monkeypatch, **fault)
    found = audit.diagnose_null(_config(tmp_path, diagnose_suites=("S1", "S3")), RunLayout(tmp_path / "null"))
    assert found["status"] == "fail" and expected in found["evidence"]["problems"], found["evidence"]["problems"]


def _fake_all(monkeypatch, *, nan=False, cases=10, excluded=0, n_paired=10, returncode=0, informational_differs=False,
              slices_differ=False, metrics=audit.SIM_HEADLINE):
    """A stand-in for ``run_all`` that writes the run report, the S0 comparison of ``metrics`` (``n_paired`` reads on
    each, or none with ``nan``), both tools' predictions and run records."""
    from alignair_assay.runners.base import RunResult, write_run_json

    def run_all(cfg, layout):
        entry = {"n_paired": 0 if nan else n_paired, "delta": None if nan else 0.0, "verdict": _TIE, "wins": 0,
                 "losses": 0, "a": None if nan else {"mean": 1.0, "n": n_paired}}
        entry["b"] = entry["a"]
        rows = {metric: dict(entry) for metric in metrics}
        info = {"v.set_size": {"a": {"mean": 1.0}, "b": {"mean": 2.0 if informational_differs else 1.0}}}
        slices = {"a": {"stratum": {"plain": 1.0}}, "b": {"stratum": {"plain": 0.5 if slices_differ else 1.0}}}
        write_json(layout.comparison_json("S0", "shared"), {"a": cfg.tools[0], "b": cfg.tools[1], "comparison": rows,
                                                            "informational": info, "by_slice": slices})
        write_json(layout.report_json, {"failed_runs": [], "suites": [{"name": "S0", "cases": cases,
                                                                       "excluded": excluded,
                                                                       "oracle": {"status": "perfect",
                                                                                  "cases": cases}}]})
        for tool in ("oracle", *cfg.tools):
            predictions = layout.predictions(tool, "S0", "shared")
            predictions.parent.mkdir(parents=True, exist_ok=True)
            predictions.write_text("sequence_id\nS0:0\n")
            write_run_json(RunResult(tool=tool, suite="S0", arm="shared", predictions=str(predictions), n_reads=1,
                                     wall_seconds=0.1, peak_rss_kb=None,
                                     returncode=returncode if tool != "oracle" else 0, commands=[], provenance={}),
                           layout.run_json(tool, "S0", "shared"))
        return 0

    monkeypatch.setattr(audit, "run_all", run_all)


_TIE = "tie_or_inconclusive"


@pytest.mark.parametrize("fault, expected", [
    ({}, None), (dict(informational_differs=True), "S0/shared v.set_size: the informational summaries differ"),
    (dict(slices_differ=True), "S0/shared: the two tools' per-slice summaries differ"),
    (dict(returncode=1), "alignair-local@a S0/shared: return code 1")])
def test_all_holds_informational_metrics_slices_and_return_codes_to_the_null(tmp_path, monkeypatch, fault, expected):
    _fake_all(monkeypatch, **fault)
    null, _ = audit.all_null(_config(tmp_path), RunLayout(tmp_path / "all"))
    if expected is None:
        assert null["status"] == "pass" and null["n"] == 10, null["reason"]
        assert null["evidence"]["runs"] == {"oracle S0/shared": 0, "alignair-local@a S0/shared": 0,
                                            "alignair-local@b S0/shared": 0}
    else:
        assert null["status"] == "fail" and expected in null["evidence"]["problems"], null["evidence"]["problems"]


def test_all_without_a_headline_metric_is_not_null(tmp_path, monkeypatch):
    # the metrics compared pair every read and differ by nothing, but clone_key.exact was never compared
    _fake_all(monkeypatch, metrics=tuple(metric for metric in audit.SIM_HEADLINE if metric != "clone_key.exact"))
    null, _ = audit.all_null(_config(tmp_path), RunLayout(tmp_path / "all"))
    assert null["status"] == "fail"
    assert "S0/shared clone_key.exact: a headline metric, not compared" in null["evidence"]["problems"]


def test_diagnose_of_a_small_model_file_against_itself_in_memory(tmp_path, make_small_model):
    # the real evaluator on S0 (10 reads): the file and the same weights in memory are two model keys
    import torch
    from alignair.core.layers import EmbeddingOrientationHead
    from alignair.model_file import save_model

    model, _ = make_small_model(REFERENCE_NAME, seed=3)
    # random weights misorient every read, and a misoriented read's boundaries are not scored: the orientation head
    # keeps every read as given (S0's with_d reads are forward), so each boundary metric pairs reads
    assert isinstance(model.orientation_head, EmbeddingOrientationHead)
    with torch.no_grad():
        model.orientation_head.fc.weight.zero_()
        model.orientation_head.fc.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # IDENTITY first
    checkpoint = tmp_path / "small.alignair"
    save_model(str(checkpoint), model, dataconfigs=[REFERENCE_NAME], training={"steps": 0, "batch_size": 0},
               include_trusted_pickle=False)
    cfg = _config(tmp_path, checkpoint=str(checkpoint))
    found = audit.diagnose_null(cfg, RunLayout(tmp_path / "null_diagnose"))
    assert found["status"] == "pass", found["reason"]
    assert found["n"] == 10 and found["evidence"]["model_key"] != found["evidence"]["baseline"]
    assert found["evidence"]["suites"]["S0"]["heads"]["metrics"] > 0


def _pair(wrong=(), nan=False):
    import math

    from alignair_assay.workflow.real_score import PairScores

    reads = [f"r{k}" for k in range(20)]
    none = math.nan if nan else None
    a = {read: {"real.v.unsupported": none if nan else 0.0, "real.v.known_top1": none if nan else 1.0,
                "real.v.ref_known_top1": none if nan else 1.0} for read in reads}
    b = {read: {"real.v.unsupported": none if nan else 1.0 if read in wrong else 0.0,
                "real.v.known_top1": none if nan else 1.0, "real.v.ref_known_top1": none if nan else 1.0}
         for read in reads}
    clusters = {read: f"{DONOR}/c{k // 2}" for k, read in enumerate(reads)}
    side = {"reads": 0, "reasons": {}, "documented": 0, "documented_wrong": 0}
    return PairScores(a=a, b=b, clusters=clusters, strata={read: "multiplex/IGHG" for read in reads},
                      excluded={"v": 0, "d": 0, "j": 0}, sensitivity_out=set(), one_sided={"a": side, "b": dict(side)},
                      cluster_sets={"sl15": clusters, "junction_length": clusters})


def _fake_real(monkeypatch, *, wrong=(), arms=("shared", "genotype"), stopped=False, summaries_differ=False,
               nan=False, uncompared=()):
    """A stand-in for ``run_real``: the checkpoint comparisons of alignair-api@a and @b on ``arms`` (and pooled),
    computed by ``real_score.comparison`` on 20 synthetic reads (@b wrong on ``wrong``; no score on either side with
    ``nan``; the ``(arm, metric)`` pairs of ``uncompared`` left out of that arm's overall comparison), and their
    predictions."""
    from alignair_assay.workflow import real_score

    seen = {}

    def run_real(cfg, layout, force=False):
        seen["cfg"] = cfg
        comparisons = []
        for arm in arms:
            pair = _pair(wrong, nan)
            entry = real_score.comparison(pair, sensitivity=["IGHV3-11"], gene_support=pair, clusterings=True,
                                          changes=True)
            for metric in (metric for where, metric in uncompared if where == arm):
                del entry["overall"][metric]
            comparisons.append({"a": cfg.tools[0], "b": cfg.tools[1], "arm": arm, "kind": "checkpoint",
                                "status": "compared", **json.loads(json.dumps(entry))})
            for tool in cfg.tools:
                path = layout.real_predictions(tool, DONOR, arm)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("sequence_id\tv_call\nr0\tIGHVF1-G1*01\n")
        tools = {arm: {tool: {"summary": {"real.v.unsupported": {"mean": 0.1 if summaries_differ and tool ==
                                                                 cfg.tools[1] else 0.0}}} for tool in cfg.tools}
                 for arm in arms}
        return {"stopped_donors": [{"donor": DONOR, "step": "preprocess", "reason": "refused"}] if stopped else [],
                "failed_runs": [], "donors": {DONOR: {"comparisons": comparisons, "tools": tools,
                                                      "oracle_self_check": {"status": "passed"}}},
                "pooled": {"comparisons": [{**entry, "donors": [DONOR]} for entry in comparisons]}}

    monkeypatch.setattr(audit, "run_real", run_real)
    return seen


def test_real_with_one_checkpoint_under_two_labels_must_compare_to_zero(tmp_path, monkeypatch):
    seen = _fake_real(monkeypatch)
    cfg = _config(tmp_path, device="cuda")
    found = audit.real_null(cfg, RunLayout(tmp_path / "null_real"))
    assert found["status"] == "pass" and found["n"] == 20, found["reason"]
    real_cfg = seen["cfg"]
    assert (real_cfg.donors, real_cfg.cap, real_cfg.tools, real_cfg.device, real_cfg.split) == (
        ("W-97",), 2000, ("alignair-api@a", "alignair-api@b"), "cuda", "dev")
    assert real_cfg.checkpoints == {"a": cfg.checkpoint, "b": cfg.checkpoint}
    assert real_cfg.data_root == cfg.data_root
    arms = found["evidence"]["arms"]
    assert set(arms) == {"shared", "genotype"} and arms["shared"]["fixed"] == arms["shared"]["broken"] == 0
    assert arms["genotype"]["predictions_identical"]


@pytest.mark.parametrize("fault, expected", [
    (dict(wrong=("r3",)), "Δ +0.05"), (dict(arms=("shared",)), "genotype: no checkpoint comparison"),
    (dict(stopped=True), "stopped at preprocess"), (dict(summaries_differ=True), "own R1 summaries differ")])
def test_real_comparisons_that_are_not_null_fail(tmp_path, monkeypatch, fault, expected):
    _fake_real(monkeypatch, **fault)
    found = audit.real_null(_config(tmp_path), RunLayout(tmp_path / "null_real"))
    assert found["status"] == "fail"
    assert any(expected in problem for problem in found["evidence"]["problems"]), found["evidence"]["problems"]


def test_real_without_a_headline_metric_on_an_arm_is_not_null(tmp_path, monkeypatch):
    # the shared arm compares every headline metric; the genotype arm lacks ref_known_top1
    _fake_real(monkeypatch, uncompared=(("genotype", "real.v.ref_known_top1"),))
    found = audit.real_null(_config(tmp_path), RunLayout(tmp_path / "null_real"))
    assert found["status"] == "fail"
    problems = found["evidence"]["problems"]
    assert f"{DONOR} genotype overall real.v.ref_known_top1: a headline metric, not compared" in problems, problems
    assert not any(problem.startswith(f"{DONOR} shared") for problem in problems)


# --- the positive control and cross-surface agreement ---------------------------------------------------------------

POSITIVE = audit.POSITIVE_METRIC


def _fake_positive(monkeypatch, layout: RunLayout, *, effect="worse", row=None, same=False):
    """Stand-ins for the perturbation, the build and ``evaluate``: the perturbed model (``kp``) against the base file
    (``kb``) on 20 reads of S0, the perturbed head metric ``effect`` (``worse``: 12 reads the base gets right are
    wrong, ``none``: no read changes, ``better``: 12 reads the base gets wrong are right), and the suite detail's
    row of that metric (default: the verdict the comparison gives, as the report renders it)."""
    import types

    from alignair_assay.diagnostics import perturb
    from alignair_assay.report.diagnostics import AGAINST_VERDICTS, against_summary
    from alignair_assay.workflow import diagnose

    seen = {}
    monkeypatch.setattr(perturb, "zero_module", lambda source, out, module: seen.update(
        perturbed=(source, Path(out), module)) or {"module": module, "parameters": ["w", "b"], "values": 10})
    monkeypatch.setattr(audit, "build_suite", lambda name, layout, **kwargs: seen.setdefault("built", []).append(name))

    def evaluate(checkpoint, run, names, **kwargs):
        seen["evaluate"] = (checkpoint, run.root, list(names), kwargs)
        base = {f"S0:{k}": {POSITIVE: 1.0 if effect != "better" or k >= 12 else 0.0} for k in range(20)}
        this = {case: {POSITIVE: 0.0 if effect == "worse" and int(case[3:]) < 12 else
                       1.0 if effect == "better" else scores[POSITIVE]} for case, scores in base.items()}
        suites = {"S0": {"cases": 20, "clusters": {"clusters": 0}, "heads": _compared(base, this), "baseline": None}}
        against = {"baseline": "kp" if same else "kb", "source": str(kwargs["against"]), "same_model": same,
                   "replay_verified": True, "suites": suites}
        summary = against_summary(against)
        verdict = AGAINST_VERDICTS[suites["S0"]["heads"]["comparison"][POSITIVE]["verdict"]]
        detail = layout.diagnostics_against_md("kp", "S0", against["baseline"])
        detail.parent.mkdir(parents=True, exist_ok=True)
        detail.write_text(f"# detail\n\n| {POSITIVE} | 100.0% | x | 20 | y | z | Δ | ci | p | {row or verdict} |\n")
        return types.SimpleNamespace(model_key="kp", against={**against, "summary": summary})

    monkeypatch.setattr(diagnose, "evaluate", evaluate)
    return seen


def test_the_positive_control_sees_the_perturbed_head_get_worse(tmp_path, monkeypatch):
    layout = RunLayout(tmp_path / "null_sim")
    cfg = _config(tmp_path, device="cuda")
    seen = _fake_positive(monkeypatch, layout)
    found = audit.positive_diagnose(cfg, layout, RunLayout(tmp_path / "positive"))
    assert found["status"] == "pass" and found["n"] == 20, found["reason"]
    source, out, module = seen["perturbed"]
    assert source == cfg.checkpoint and out.parent == tmp_path / "positive" and module == audit.POSITIVE_MODULE
    checkpoint, root, names, kwargs = seen["evaluate"]
    # the perturbed checkpoint is the model under test, the audited one the baseline, on the null's shared build
    assert checkpoint == str(out) and kwargs["against"] == cfg.checkpoint and root == layout.root and names == ["S0"]
    assert kwargs["profile"] is False and kwargs["batch_size"] == cfg.batch_size
    evidence = found["evidence"]
    assert evidence["delta"] == pytest.approx(-0.6) and evidence["verdict"] == "a_better"
    assert (evidence["fixed"], evidence["broken"]) == (0, 12)
    assert "baseline better" in evidence["summary"]


@pytest.mark.parametrize("fault, expected", [
    (dict(effect="none"), f"Δ 0 on {POSITIVE}"),
    (dict(effect="better"), f"Δ +0.6 on {POSITIVE}"),
    (dict(row="this better"), f"the suite detail's {POSITIVE} row does not read 'baseline better'"),
    (dict(same=True), "one model key"),
], ids=["no_effect", "wrong_direction", "rendered_direction", "one_key"])
def test_a_positive_control_that_sees_no_damage_or_the_wrong_direction_fails(tmp_path, monkeypatch, fault, expected):
    layout = RunLayout(tmp_path / "null_sim")
    _fake_positive(monkeypatch, layout, **fault)
    found = audit.positive_diagnose(_config(tmp_path), layout, RunLayout(tmp_path / "positive"))
    assert found["status"] == "fail"
    assert any(expected in problem for problem in found["evidence"]["problems"]), found["evidence"]["problems"]


def _diagnosed(layout: RunLayout, key: str, suite: str, baseline: dict) -> None:
    """A diagnose record of ``suite`` for model ``key`` in ``layout``: its per-case scores (``baseline`` per case,
    None for an excluded case) and the record naming them and the suite's build files."""
    from alignair_assay.core.provenance import sha256_file
    from alignair_assay.workflow.diagnose import CASES_KIND

    def encoded(scores):
        return None if scores is None else {name: None if value != value else value for name, value in scores.items()}

    cases = layout.diagnostics_suite_cases(key, suite)
    cases.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cases, [{"case_id": case, "head": {}, "baseline": encoded(scores), "variants": {}}
                        for case, scores in baseline.items()], header={"kind": CASES_KIND})
    write_json(layout.diagnostics_suite_json(key, suite),
               {"key": {"suite": {"name": suite, "cases_sha256": sha256_file(layout.cases(suite)),
                                  "truth_sha256": sha256_file(layout.truth(suite))}},
                "case_scores": {"sha256": sha256_file(cases)}})


@pytest.fixture
def surfaces(tmp_path, monkeypatch):
    """One S0 build's per-case scores on both surfaces: all's alignair-local@b and diagnose's baseline for the
    checkpoint's model key (``kb``), equal on 6 cases (one excluded on both, one NaN on both)."""
    import math

    layout = RunLayout(tmp_path / "null_sim")
    layout.suite_dir("S0").mkdir(parents=True)
    layout.cases("S0").write_text("cases\n")
    layout.truth("S0").write_text("truth\n")
    scores = {f"S0:{k}": {"v.top1_in_set": 1.0, "junction.nt_exact": math.nan if k == 2 else 0.0}
              for k in range(6)}
    layout.scores_dir("S0", "shared").mkdir(parents=True)
    write_scores(scores, layout.tool_scores("S0", "shared", "alignair-local@b"))
    _diagnosed(layout, "kb", "S0", {**{case: {**value, "head.extra": 1.0} for case, value in scores.items()},
                                    "S0:9": None})
    monkeypatch.setattr(audit, "checkpoint_model_key", lambda checkpoint: "kb")
    return layout


def test_all_and_diagnose_score_the_checkpoint_alike_case_by_case(tmp_path, surfaces):
    found = audit.all_vs_diagnose(_config(tmp_path), surfaces)
    assert found["status"] == "pass" and found["n"] == 6, found["reason"]
    evidence = found["evidence"]
    assert evidence["metrics"] == ["junction.nt_exact", "v.top1_in_set"] and evidence["model_key"] == "kb"
    assert evidence["tool"] == "alignair-local@b" and evidence["suite"] == "S0" and evidence["differing"] == 0


@pytest.mark.parametrize("fault, expected", [
    ("value", "S0:3 v.top1_in_set: all 0, diagnose 1"),
    ("nan", "S0:4 junction.nt_exact: all nan, diagnose 0"),
    ("case", "cases only all scores: 1 (e.g. S0:5)"),
    ("build", "diagnosed on another build of S0"),
    ("missing", "no diagnose record of S0 for the checkpoint (kb)"),
], ids=["value", "nan", "case", "build", "missing"])
def test_all_and_diagnose_that_disagree_fail(tmp_path, surfaces, fault, expected):
    import math

    from alignair_assay.scoring.pipeline import read_scores

    path = surfaces.tool_scores("S0", "shared", "alignair-local@b")
    scores = read_scores(path)
    if fault == "value":
        scores["S0:3"]["v.top1_in_set"] = 0.0
        write_scores(scores, path)
    elif fault == "nan":
        scores["S0:4"]["junction.nt_exact"] = math.nan
        write_scores(scores, path)
    elif fault == "case":
        _diagnosed(surfaces, "kb", "S0", {case: value for case, value in scores.items() if case != "S0:5"})
    elif fault == "build":
        surfaces.cases("S0").write_text("rebuilt\n")
    else:
        surfaces.diagnostics_suite_json("kb", "S0").unlink()
    found = audit.all_vs_diagnose(_config(tmp_path), surfaces)
    assert found["status"] == "fail"
    assert any(expected in problem for problem in found["evidence"]["problems"]), found["evidence"]["problems"]


# --- speed --------------------------------------------------------------------------------------------------------

STORED_SPEED = {"results": {
    "igblast_1t": {"tool": "igblast", "threads": 1, "steady_reads_per_s": 62.8, "measured_utc": "2026-10-20",
                   "settings": {"igblastn_sha256": "x"}},
    "igblast_8t": {"tool": "igblast", "threads": 8, "steady_reads_per_s": 398.5, "measured_utc": "2026-10-20",
                   "settings": {"igblastn_sha256": "x"}},
    "alignair_8t": {"tool": "alignair", "threads": 8, "steady_reads_per_s": 113.4}}}


def _fake_speed(monkeypatch, rates: dict):
    """Stand-ins for the S1 build and ``run_speed``, which measures ``rates`` (config -> reads/s, or an error)."""
    seen = {}
    monkeypatch.setattr(audit, "build_suite", lambda name, layout, **kwargs: seen.setdefault("built", (name, kwargs)))
    monkeypatch.setattr(audit, "truth_reference", lambda layout, names: "reference")

    def run_speed(cfg, layout, reference, *, names=None, **kwargs):
        seen["run"] = (cfg, list(names))
        results = {name: ({"error": rate} if isinstance(rate, str) else
                          {"steady_reads_per_s": rate, "settings": {"igblastn_sha256": "x"}})
                   for name, rate in rates.items()}
        return {"n_full": 4000, "repeats": 3, "results": results}

    monkeypatch.setattr(audit, "run_speed", run_speed)
    return seen


def _speed_run(tmp_path) -> RunLayout:
    sim = _sim_run(tmp_path / "sim", suites=("S1",), speed=STORED_SPEED)
    (tmp_path / "sim" / "igblast").mkdir()
    return sim


def test_igblast_speed_within_fifteen_percent_of_the_stored_measure_passes(tmp_path, monkeypatch):
    seen = _fake_speed(monkeypatch, {"igblast_1t": 60.0, "igblast_8t": 420.0})
    sim = _speed_run(tmp_path)
    found = audit.igblast_speed(_config(tmp_path), sim, RunLayout(tmp_path / "speed"), load=lambda: 0.4,
                                sleep=lambda seconds: None)
    assert found["status"] == "pass" and found["n"] == 2, found["reason"]
    cfg, names = seen["run"]
    assert names == ["igblast_1t", "igblast_8t"] and cfg.tools == ("igblast",) and cfg.threads == 8
    assert cfg.igblast_dir == str(tmp_path / "sim" / "igblast")  # the release the stored run recorded
    assert seen["built"][0] == "S1" and seen["built"][1]["split"] == "dev"  # never the locked test split
    assert found["evidence"]["configs"]["igblast_8t"]["ratio"] == pytest.approx(420.0 / 398.5)
    assert "igblast_8t 420.0 reads/s vs 398.5 reads/s stored (+5%)" in found["evidence"]["summary"]


def test_igblast_speed_outside_the_band_or_failing_warns_and_never_fails(tmp_path, monkeypatch):
    sim = _speed_run(tmp_path)
    for rates in ({"igblast_1t": 60.0, "igblast_8t": 300.0}, {"igblast_1t": 60.0, "igblast_8t": "RuntimeError: x"}):
        _fake_speed(monkeypatch, rates)
        found = audit.igblast_speed(_config(tmp_path), sim, RunLayout(tmp_path / "speed"), load=lambda: 0.4,
                                    sleep=lambda seconds: None)
        assert found["status"] == "warn" and "igblast_8t" in found["reason"] and "never fails" in found["reason"]


def test_a_busy_machine_skips_the_speed_check(tmp_path, monkeypatch):
    seen = _fake_speed(monkeypatch, {"igblast_1t": 60.0, "igblast_8t": 400.0})
    sim = _speed_run(tmp_path)
    naps = []
    found = audit.igblast_speed(_config(tmp_path), sim, RunLayout(tmp_path / "speed"), load=lambda: 3.2,
                                sleep=naps.append)
    assert found["status"] == "skipped" and found["reason"].startswith("skipped (load)")
    assert "built" not in seen and "run" not in seen  # nothing timed, nothing built
    assert sum(naps) >= audit.SPEED_SETTLE_SECONDS and found["evidence"]["load"] == 3.2
    # a load that settles after the build: timed
    loads = iter([0.5, 2.5, 1.4, 1.0])
    found = audit.igblast_speed(_config(tmp_path), sim, RunLayout(tmp_path / "speed"), load=lambda: next(loads),
                                sleep=lambda seconds: None)
    assert found["status"] == "pass" and found["evidence"]["load_before"] == 1.4


def test_an_error_while_timing_igblast_warns(tmp_path, monkeypatch):
    _fake_speed(monkeypatch, {})

    def broken(*args, **kwargs):
        raise RuntimeError("makeblastdb failed")

    monkeypatch.setattr(audit, "run_speed", broken)
    found = audit.igblast_speed(_config(tmp_path), _speed_run(tmp_path), RunLayout(tmp_path / "speed"),
                                load=lambda: 0.4, sleep=lambda seconds: None)
    assert found["status"] == "warn" and "RuntimeError: makeblastdb failed" in found["reason"]
    assert "informational" in found["reason"]


def test_speed_is_skipped_when_asked_or_without_a_stored_measure(tmp_path, monkeypatch):
    seen = _fake_speed(monkeypatch, {})
    found = audit.igblast_speed(_config(tmp_path, skip_speed=True), _speed_run(tmp_path), RunLayout(tmp_path / "s"))
    assert found["status"] == "skipped" and "--skip-speed" in found["reason"]
    bare = _sim_run(tmp_path / "bare", suites=("S1",))
    found = audit.igblast_speed(_config(tmp_path), bare, RunLayout(tmp_path / "s"), load=lambda: 0.1)
    assert found["status"] == "skipped" and "no current IgBLAST speed" in found["reason"]
    stale = _sim_run(tmp_path / "stale", suites=("S1",), speed={"stale": True, "reason": "another configuration"})
    assert audit.igblast_speed(_config(tmp_path), stale, RunLayout(tmp_path / "s"))["status"] == "skipped"
    assert seen == {}


# --- IgBLAST's S1 V top-1 at this code ----------------------------------------------------------------------------

class FakeIgblast(OracleRunner):
    """IgBLAST as the oracle, with the V call of the first ``wrong`` reads made wrong."""

    name = "igblast"
    wrong = 0
    prepared: list = []

    def prepare(self, layout, reference):
        FakeIgblast.prepared.append(layout.root)
        return super().prepare(layout, reference)

    def run(self, fasta, out_tsv, **kwargs):
        result = super().run(fasta, out_tsv, **kwargs)
        lines = Path(out_tsv).read_text().splitlines()
        column = lines[0].split("\t").index("v_call")
        for row in range(1, 1 + FakeIgblast.wrong):
            cells = lines[row].split("\t")
            cells[column] = "IGHV1-2*99"
            lines[row] = "\t".join(cells)
        Path(out_tsv).write_text("\n".join(lines) + "\n")
        return dataclasses.replace(result, tool=self.name)


@pytest.fixture
def fake_igblast(monkeypatch):
    FakeIgblast.wrong, FakeIgblast.prepared = 0, []
    seen = {}

    def make(tool, options):
        seen["options"] = (tool, options)
        return FakeIgblast()

    monkeypatch.setattr(audit, "make_runner", make)
    return seen


@pytest.fixture(scope="module")
def s1_build(tmp_path_factory):
    """S1 built once on the dev split at scale 0.02 (100 cases, with donors): copied where a test needs it."""
    layout = RunLayout(tmp_path_factory.mktemp("s1"))
    build_suite("S1", layout, reference_name=REFERENCE_NAME, split="dev", seed=7, scale=0.02, workers=4)
    return layout


def _with_s1(s1_build: RunLayout, root: Path) -> RunLayout:
    """A run directory holding a copy of ``s1_build``'s S1."""
    import shutil

    shutil.copytree(s1_build.suite_dir("S1"), RunLayout(root).suite_dir("S1"))
    return RunLayout(root)


def test_igblast_s1_v_top1_at_this_code_within_one_and_a_half_points_passes(tmp_path, fake_igblast, s1_build,
                                                                             monkeypatch):
    sim = _speed_run(tmp_path)
    builds = []

    def build(name, layout, **kwargs):  # S1 at the given scale, as the module's build of it
        builds.append((name, layout.root, kwargs["scale"], kwargs["split"]))
        _with_s1(s1_build, layout.root)

    monkeypatch.setattr(audit, "build_suite", build)
    FakeIgblast.wrong = 2  # 98 of 100 right
    cfg = _config(tmp_path, v_top1_scale=0.02)
    found = audit.igblast_v_top1_current(cfg, sim, RunLayout(tmp_path / "speed"), RunLayout(tmp_path / "own"))
    assert found["status"] == "pass" and found["n"] == 100, found["reason"]
    assert found["evidence"]["mean"] == pytest.approx(0.98) and found["evidence"]["split"] == "dev"
    assert "S1 v.top1_in_set 98.00% (n=100)" in found["evidence"]["summary"]
    # no S1 from the speed check: S1 built in its own scratch, on the dev split, at the given scale
    assert builds == [("S1", tmp_path / "own", 0.02, "dev")] and FakeIgblast.prepared == [tmp_path / "own"]
    tool, options = fake_igblast["options"]
    assert tool == "igblast" and options.igblast_dir == str(tmp_path / "sim" / "igblast")


def test_igblast_s1_v_top1_at_this_code_outside_the_band_fails(tmp_path, fake_igblast, s1_build):
    # the gate: IgBLAST's S1 V top-1 at this code, on fresh dev reads
    assert "band.sim_v_top1_current" in audit.GATING
    sim = _speed_run(tmp_path)
    speed = _with_s1(s1_build, tmp_path / "speed")
    for wrong in (0, 5):  # 100%, 95%: both outside 97.9% ± 1.5 pp
        FakeIgblast.wrong = wrong
        found = audit.igblast_v_top1_current(_config(tmp_path), sim, speed, RunLayout(tmp_path / "own"))
        assert found["status"] == "fail" and "outside 97.9% ± 1.5 pp" in found["reason"], found


def test_igblast_s1_v_top1_at_this_code_reuses_the_speed_checks_s1(tmp_path, fake_igblast, s1_build, monkeypatch):
    sim = _speed_run(tmp_path)
    speed = _with_s1(s1_build, tmp_path / "speed")
    builds = []
    monkeypatch.setattr(audit, "build_suite", lambda *args, **kwargs: builds.append(args))
    FakeIgblast.wrong = 2
    found = audit.igblast_v_top1_current(_config(tmp_path), sim, speed, RunLayout(tmp_path / "own"))
    assert found["status"] == "pass" and builds == [] and FakeIgblast.prepared == [speed.root]
    assert found["evidence"]["reused_speed_build"]


def test_igblast_s1_v_top1_at_this_code_raises_or_is_skipped_without_igblast(tmp_path, fake_igblast, s1_build,
                                                                             monkeypatch):
    # a gating check that raises fails the audit (run_audit records the error); without IgBLAST it is skipped, which
    # leaves the audit incomplete
    sim = _speed_run(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("igblastn crashed")

    monkeypatch.setattr(audit, "run_suite", broken)
    with pytest.raises(RuntimeError, match="igblastn crashed"):
        audit.igblast_v_top1_current(_config(tmp_path), sim, _with_s1(s1_build, tmp_path / "speed"),
                                     RunLayout(tmp_path / "own"))
    bare = _sim_run(tmp_path / "bare", suites=("S1",))  # its recorded IgBLAST release does not exist
    found = audit.igblast_v_top1_current(_config(tmp_path), bare, RunLayout(tmp_path / "speed"),
                                         RunLayout(tmp_path / "own"))
    assert found["status"] == "skipped" and "IgBLAST release" in found["reason"]
    assert verdict([check("null.all", "pass", 10, {}), found])["status"] == "incomplete"


# --- the audit ----------------------------------------------------------------------------------------------------

def _stored_runs(tmp_path) -> AuditConfig:
    _sim_run(tmp_path / "sim", suites=("S1",))
    _real_run(tmp_path / "real", {"W-97": _donor()})
    return _config(tmp_path)


# The (run directory, check) pairs the fake checks were given, first run directory argument after the config.
_CALLS: list = []


def _fake_checks(monkeypatch, *, raising: str | None = None, statuses: dict | None = None) -> list[Path]:
    """Every check function replaced by one returning ``statuses[name]`` (default pass, n 10), or raising for the
    check ``raising``; returns the run directories (``RunLayout`` roots) each was given, in call order."""
    statuses = statuses or {}
    roots = []
    calls = _CALLS
    calls.clear()

    def fake(*names):
        def run(*args):
            roots.extend(Path(arg.root) for arg in args if isinstance(arg, RunLayout))
            calls.extend((Path(arg.root), names[0]) for arg in args[1:2] if isinstance(arg, RunLayout))
            if raising in names:
                raise RuntimeError("docker is not running")
            found = [check(name, statuses.get(name, "pass"), 10, {"summary": f"{name} checked"},
                           None if statuses.get(name, "pass") == "pass" else "why") for name in names]
            return found if len(found) > 1 else found[0]
        return run

    for function, names in (("igblast_speed", ("band.speed",)),
                            ("igblast_v_top1_current", ("band.sim_v_top1_current",)),
                            ("sim_oracle", ("self_check.sim_oracle",)),
                            ("sim_staleness", ("self_check.sim_staleness",)),
                            ("real_oracle", ("self_check.real_oracle",)),
                            ("real_truth_health", ("self_check.real_truth_health",)),
                            ("real_known_band", ("band.real_known",)), ("sim_v_top1_band", ("band.sim_v_top1",)),
                            ("sim_reproducible", ("reproducible.sim_build",)),
                            ("real_reproducible", ("reproducible.real_scores",)),
                            ("all_null", ("null.all", "self_check.sim_oracle_current")),
                            ("diagnose_null", ("null.diagnose",)), ("positive_diagnose", ("positive.diagnose",)),
                            ("all_vs_diagnose", ("agreement.all_vs_diagnose",)), ("real_null", ("null.real",))):
        monkeypatch.setattr(audit, function, fake(*names))
    return roots


def test_the_audit_runs_every_check_and_writes_its_record(tmp_path, monkeypatch, capsys):
    from alignair_assay.core.provenance import CODE_SUFFIXES, tree_digest

    cfg = _stored_runs(tmp_path)
    roots = _fake_checks(monkeypatch, statuses={"self_check.real_truth_health": "warn"})
    calls = _CALLS
    layout = RunLayout(tmp_path / "out")
    stale = layout.audit_scratch("null_sim") / "old.txt"  # an earlier audit's scratch is replaced
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    record = audit.run_audit(cfg, layout)
    assert [found["name"] for found in record["checks"]] == list(CHECKS)
    assert all(set(found) == {"name", "status", "n", "evidence", "reason"} for found in record["checks"])
    assert all(found["evidence"]["seconds"] >= 0 for found in record["checks"])
    assert record["verdict"]["status"] == "pass" and record["verdict"]["counts"]["warn"] == 1
    assert json.loads(layout.audit_json.read_text()) == record
    assert layout.audit_md.read_text().splitlines()[0] == record["verdict"]["line"]
    assert record["checkpoint"]["path"] == cfg.checkpoint and len(record["checkpoint"]["sha256"]) == 64
    assert not stale.exists()
    # every run directory a check was given is a scratch directory of the audit's own or a stored run (read only)
    scratches = [root for root in roots if layout.audit_dir in root.parents]
    assert set(roots) - set(scratches) == {Path(cfg.sim_run), Path(cfg.real_run)}
    assert scratches[0] == layout.audit_scratch("speed")  # IgBLAST is timed first, on a quiet machine
    assert {root.name for root in scratches} == {"speed", "sim_v_top1_current", "sim_build_1", "sim_build_2",
                                                 "real_scores", "null_sim", "positive", "null_real"}
    # all, diagnose, the positive control and the agreement share one dev build: they are given one directory
    given = {name: [root for root, check_name in calls if check_name == name] for name in (
        "null.all", "null.diagnose", "positive.diagnose", "agreement.all_vs_diagnose")}
    assert {tuple(roots) for roots in given.values()} == {(layout.audit_scratch("null_sim"),)}
    # the record names the assay code it ran (its sources' digest), and the markdown shows it
    package = Path(audit.__file__).resolve().parents[1]
    code = tree_digest(package, suffixes=CODE_SUFFIXES)
    assert record["provenance"]["audit_code_sha256"] == code and f"`{code[:12]}`" in layout.audit_md.read_text()
    # the GenAIRR it ran: the frozen copy's digest, and what was frozen (the tests run a frozen copy)
    import os

    from alignair_assay.core.freeze import LIVE_ENV, frozen_record

    genairr = record["provenance"]["genairr"]
    assert record["provenance"]["genairr_freeze"] == frozen_record(genairr["genairr_source"])
    if os.environ.get(LIVE_ENV) != "1":
        assert record["provenance"]["genairr_freeze"]["genairr_source_sha256"] == genairr["genairr_source_sha256"]
        assert "a frozen copy of commit" in layout.audit_md.read_text()
    # each check is printed once, as it finishes
    printed = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in printed if line.startswith("audit ")] == [
        f"audit {name}" for name in ("band.speed", "band.sim_v_top1_current", "self_check.sim_oracle",
                                     "self_check.sim_staleness", "self_check.real_oracle",
                                     "self_check.real_truth_health", "band.real_known", "band.sim_v_top1",
                                     "reproducible.sim_build", "reproducible.real_scores", "null.all",
                                     "self_check.sim_oracle_current", "null.diagnose", "positive.diagnose",
                                     "agreement.all_vs_diagnose", "null.real")]


def test_the_audit_names_the_code_it_started_with(tmp_path, monkeypatch):
    # the whole package's digest is taken before any check runs (code edited meanwhile does not rename the run), under
    # a name of its own: diagnose's assay_code_sha256 digests other files
    cfg = _stored_runs(tmp_path)
    roots = _fake_checks(monkeypatch)
    monkeypatch.setattr(audit, "tree_digest", lambda *args, **kwargs: "after a check" if roots else "before")
    record = audit.run_audit(cfg, RunLayout(tmp_path / "out"))
    assert record["provenance"]["audit_code_sha256"] == "before"
    assert "assay_code_sha256" not in record["provenance"]


def test_a_check_that_raises_fails_and_the_others_go_on(tmp_path, monkeypatch):
    cfg = _stored_runs(tmp_path)
    _fake_checks(monkeypatch, raising="null.all")
    record = audit.run_audit(cfg, RunLayout(tmp_path / "out"))
    found = {entry["name"]: entry for entry in record["checks"]}
    for name in ("null.all", "self_check.sim_oracle_current"):
        assert found[name]["status"] == "fail" and found[name]["reason"] == "RuntimeError: docker is not running"
    assert found["null.real"]["status"] == "pass" and len(found) == len(CHECKS)
    assert record["verdict"]["status"] == "fail" and record["verdict"]["failed"] == ["self_check.sim_oracle_current",
                                                                                      "null.all"]


def test_an_informational_check_that_raises_warns(tmp_path, monkeypatch):
    # speed is informational: whatever goes wrong while timing IgBLAST, it never fails the audit
    cfg = _stored_runs(tmp_path)
    _fake_checks(monkeypatch, raising="band.speed")
    record = audit.run_audit(cfg, RunLayout(tmp_path / "out"))
    speed = next(entry for entry in record["checks"] if entry["name"] == "band.speed")
    assert speed["status"] == "warn" and "docker is not running" in speed["reason"]
    assert record["verdict"]["status"] == "pass"


def test_the_audit_refuses_inputs_before_any_work(tmp_path, monkeypatch):
    cfg = _stored_runs(tmp_path)
    roots = _fake_checks(monkeypatch)
    for root in (tmp_path / "sim", tmp_path / "sim" / "inside", tmp_path / "real", tmp_path):
        with pytest.raises(ValueError, match="overlaps the stored run"):
            audit.run_audit(cfg, RunLayout(root))
    with pytest.raises(FileNotFoundError, match="no AlignAIR checkpoint"):
        audit.run_audit(dataclasses.replace(cfg, checkpoint=str(tmp_path / "gone.alignair")),
                        RunLayout(tmp_path / "out"))
    with pytest.raises(FileNotFoundError, match="made by `real`"):
        audit.run_audit(dataclasses.replace(cfg, real_run=str(tmp_path / "sim")), RunLayout(tmp_path / "out"))
    with pytest.raises(FileNotFoundError, match="made by `all`"):
        audit.run_audit(dataclasses.replace(cfg, sim_run=str(tmp_path / "real")), RunLayout(tmp_path / "out"))
    assert roots == [] and not (tmp_path / "out").exists()


def test_the_audit_command(tmp_path, monkeypatch, capsys):
    cfg = _stored_runs(tmp_path)
    seen = {}

    def run_audit(given, layout):
        seen["cfg"], seen["layout"] = given, layout
        checks = [check("null.all", seen.get("status", "pass"), 10, {"summary": "zero"}, "x")]
        return _record(checks)

    monkeypatch.setattr(audit, "run_audit", run_audit)
    argv = ["audit", "--sim-run", cfg.sim_run, "--real-run", cfg.real_run, "--data-root", cfg.data_root,
            "--checkpoint", f"base={cfg.checkpoint}", "--device", "cuda", "--skip-speed", "--out",
            str(tmp_path / "out")]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "AUDIT PASS: 1 checks" in out and str(RunLayout(tmp_path / "out").audit_md) in out
    assert "null.all" not in out  # the workflow prints each check as it finishes; the command adds the verdict only
    given = seen["cfg"]
    assert (given.checkpoint, given.device, given.skip_speed) == (cfg.checkpoint, "cuda", True)
    assert seen["layout"].root == tmp_path / "out"
    seen["status"] = "fail"
    assert main(argv) == 2 and "AUDIT FAIL" in capsys.readouterr().out
    seen["status"] = "skipped"  # null.all gates: skipped, the audit is incomplete
    assert main(argv) == 3 and "AUDIT INCOMPLETE" in capsys.readouterr().out
    # one checkpoint, labelled base
    assert argv[7:9] == ["--checkpoint", f"base={cfg.checkpoint}"]
    assert main([*argv[:7], "--checkpoint", f"cand={cfg.checkpoint}", *argv[9:]]) == 2
    assert "--checkpoint base=PATH" in capsys.readouterr().err
    # run on the live GenAIRR, it says so
    from alignair_assay.cli import audit as audit_command

    monkeypatch.setattr(audit_command, "frozen_record", lambda source: None)
    seen["status"] = "pass"
    assert main(argv) == 0 and "warning: GenAIRR is the live checkout" in capsys.readouterr().err
