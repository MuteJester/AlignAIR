"""``show``: one read looked up in a run directory, side by side across tools (``workflow.show``, ``report.show``,
the ``show`` command).

Simulated (S0 built at scale 0.2: 10 cases): the case, its observable truth, each tool's AIRR row and scores (two
tools answering with the oracle's rows, one with another V call on one case), with the metrics where they differ
marked, and each diagnosed model's stored per-case scores. Real (a hand-made donor): the read, its truth, each
tool's AIRR row, R1 class and scores. A pure lookup: nothing is written. An unknown id is refused naming the three
nearest.
"""
import csv
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.core.records import write_json, write_jsonl  # noqa: E402
from alignair_assay.real.truth import RealTruth, truth_to_json  # noqa: E402
from alignair_assay.runners.oracle import OracleRunner  # noqa: E402
from alignair_assay.scoring.pipeline import truth_reference, write_scores  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from alignair_assay.workflow.scoring import score_file  # noqa: E402
from alignair_assay.workflow.show import show_case, show_read  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
REFERENCE = "HUMAN_IGH_OGRDB"
TOOLS = ("igblast", "alignair-local")
KEYS = ("aaaaaaaaaaaa-111111111111", "bbbbbbbbbbbb-111111111111")


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _diagnosed(layout: RunLayout, key: str, suite: str, lines: list[dict]) -> None:
    """A diagnose record of ``suite`` for model ``key``: its per-case scores and the record naming them."""
    cases = layout.diagnostics_suite_cases(key, suite)
    cases.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cases, lines, header={"kind": "alignair_assay.diagnostics_cases"})
    write_json(layout.diagnostics_suite_json(key, suite), {"case_scores": {"sha256": sha256_file(cases)}})


@pytest.fixture(scope="module")
def sim_run(tmp_path_factory):
    """S0 built on the dev split, two tools' predictions and scores on its shared arm (``alignair-local`` calls the
    first case's V otherwise) and two diagnosed models' per-case scores (the second one's V top-1 head wrong on it)."""
    layout = RunLayout(tmp_path_factory.mktemp("show") / "run")
    build_suite("S0", layout, reference_name=REFERENCE, split="dev", seed=7, scale=0.2, workers=1)
    reference = truth_reference(layout, ["S0"])
    truths = read_truth_jsonl(layout.truth("S0"))
    first = truths[0].case_id
    oracle = OracleRunner(layout)
    for tool in TOOLS:
        out = layout.predictions(tool, "S0", "shared")
        out.parent.mkdir(parents=True)
        oracle.run(layout.reads("S0"), out, suite="S0", arm="shared", log_dir=out.parent)
    rows = _rows(layout.predictions("alignair-local", "S0", "shared"))
    truth_v = set(truths[0].segments["v"].alleles.alleles)
    other = next(name for name in sorted(reference.alleles["v"]) if name not in truth_v)
    rows[0]["v_call"] = other
    _write_rows(layout.predictions("alignair-local", "S0", "shared"), rows)
    for tool in TOOLS:
        scored = score_file(tool, truths, layout.predictions(tool, "S0", "shared"), reference)
        write_scores(scored.scores, layout.tool_scores("S0", "shared", tool))
    for key, right in zip(KEYS, (1.0, 0.0)):
        _diagnosed(layout, key, "S0", [{"case_id": truth.case_id,
                                        "head": {"head.v.allele_top1": right if truth.case_id == first else 1.0},
                                        "baseline": {"v.top1_in_set": 1.0}, "variants": {}} for truth in truths])
    return layout, first, other


def _snapshot(root: Path) -> dict:
    return {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in sorted(root.rglob("*"))}


def test_a_simulated_case_side_by_side(sim_run, capsys):
    layout, first, other = sim_run
    before = _snapshot(layout.root)
    assert main(["show", "--run-dir", str(layout.root), "--case", first]) == 0
    out = capsys.readouterr().out
    assert _snapshot(layout.root) == before  # a pure lookup: nothing written
    case = read_cases(layout.cases("S0"))[0]
    truth = read_truth_jsonl(layout.truth("S0"))[0]
    assert first in out and case["sequence"] in out and case["stratum"] in out
    assert f"simulated calls: V {case['record']['v_call']}" in out
    # the truth: the scored allele set and spans of each gene
    assert "## Observable truth" in out and sorted(truth.segments["v"].alleles.alleles)[0] in out
    # each tool's AIRR row, side by side
    assert "| Field | alignair-local | igblast |" in out
    v_row = next(line for line in out.splitlines() if line.startswith("| v_call |"))
    assert other in v_row and sorted(truth.segments["v"].alleles.alleles)[0] in v_row
    # each tool's scores, the differing metrics marked
    assert "| Metric | alignair-local | igblast | Differs |" in out
    assert next(line for line in out.splitlines() if line.startswith("| v.top1_in_set |")).endswith("| ≠ |")
    assert next(line for line in out.splitlines() if line.startswith("| j.top1_in_set |")).endswith("|  |")
    # the diagnosed models' stored scores, read without a replay
    assert f"| Metric | {KEYS[0]} | {KEYS[1]} | Differs |" in out
    assert next(line for line in out.splitlines() if line.startswith("| head.v.allele_top1 |")).endswith("| ≠ |")


def test_the_lookup_returns_the_data_it_shows(sim_run):
    layout, first, other = sim_run
    found = show_case(layout, first, tools=["alignair-local"])
    assert found["kind"] == "sim" and found["suite"] == "S0" and found["arm"] == "shared"
    assert list(found["tools"]) == ["alignair-local"] and found["tools"]["alignair-local"]["row"]["v_call"] == other
    assert found["tools"]["alignair-local"]["scores"]["v.top1_in_set"] == 0.0
    assert found["truth"]["case_id"] == first and found["case"]["simulated"]["v_call"]
    assert set(found["diagnose"]) == set(KEYS) and all(entry["current"] for entry in found["diagnose"].values())
    assert found["diagnose"][KEYS[1]]["head"]["head.v.allele_top1"] == 0.0
    # every tool that ran by default; a tool that did not run is refused naming those that did
    assert list(show_case(layout, first)["tools"]) == sorted(TOOLS)
    with pytest.raises(ValueError, match=r"no predictions of tool 'nope' on S0/shared; tools that ran: "
                                         r"alignair-local, igblast"):
        show_case(layout, first, tools=["nope"])


def test_an_unknown_case_is_refused_with_the_three_nearest(sim_run, capsys):
    layout, first, _ = sim_run
    wrong = first[:-1] + ("9" if first[-1] != "9" else "8")
    assert main(["show", "--run-dir", str(layout.root), "--case", wrong]) == 2
    err = capsys.readouterr().err
    assert f"no case {wrong!r}" in err and "the nearest:" in err
    nearest = err.split("the nearest:")[1].strip().split(", ")
    assert len(nearest) == 3 and all(name.startswith("S0:") for name in nearest)
    assert main(["show", "--run-dir", str(layout.root), "--case", "S9:full:000000"]) == 2
    assert "the nearest:" in capsys.readouterr().err


@pytest.fixture
def real_run(tmp_path):
    """A real run directory with donor W-97: three reads, their truths, two tools' rows and per-read classes."""
    layout = RunLayout(tmp_path / "real_run")
    donor = layout.real_donor_dir("W-97")
    donor.mkdir(parents=True)
    reads = {"SRR1.1": "ACGTACGTAC", "SRR1.2": "TTTTGGGGCC", "SRR1.30": "GGGGAAAATT"}
    layout.real_reads("W-97").write_text("".join(f">{name}\n{seq}\n" for name, seq in reads.items()))
    layout.real_reads_meta("W-97").write_text("sequence_id\tisotype\tprotocol\tconsensus_count\tdupcount\t"
                                              "primer_masked\n" + "".join(
                                                  f"{name}\tIGHM\tmultiplex\t1\t1\t[[0,2]]\n" for name in reads))
    truths = [RealTruth(read_id=name, donor="W-97", isotype="IGHM", protocol="multiplex", observed=(2, 10),
                        known=frozenset({"IGHV3-23*01"}) if name == "SRR1.2" else None,
                        known_shared=frozenset({"IGHV3-23*01"}) if name == "SRR1.2" else None,
                        known_gene="IGHV3-23" if name == "SRR1.2" else None, novel_known=False, naive=False,
                        clonotype=f"W-97/single:{name}", anchor="donor", cys_end=9)
              for name in reads]
    write_jsonl(layout.real_truth("W-97"), [truth_to_json(truth) for truth in truths], header={"kind": "truth"})
    calls = {"igblast": "IGHV3-23*01", "alignair-api": "IGHV3-30*02"}
    for tool, call in calls.items():
        _write_rows(layout.real_predictions(tool, "W-97", "shared"),
                    [{"sequence_id": name, "v_call": call, "d_call": "", "j_call": "IGHJ4*02", "productive": "T"}
                     for name in reads])
        path = layout.real_scores("W-97", "shared", tool)
        path.parent.mkdir(parents=True, exist_ok=True)
        support = "supported" if tool == "igblast" else "unsupported"
        write_jsonl(path, [{"read_id": name, "v_call": call, "v": support, "d": None, "j": None, "any": True,
                            "v_support": support} for name in reads], header={"kind": "scores"})
    return layout


def test_a_real_read_side_by_side(real_run, capsys):
    before = _snapshot(real_run.root)
    assert main(["show", "--run-dir", str(real_run.root), "--donor", "W-97", "--read", "SRR1.2"]) == 0
    out = capsys.readouterr().out
    assert _snapshot(real_run.root) == before
    assert "SRR1.2" in out and "TTTTGGGGCC" in out and "multiplex" in out and "IGHM" in out
    assert "## Truth" in out and "known: IGHV3-23*01" in out and "clonotype: W-97/single:SRR1.2" in out
    assert "| Field | alignair-api | igblast |" in out
    r1 = next(line for line in out.splitlines() if line.startswith("| R1 class (v) |"))
    assert r1 == "| R1 class (v) | unsupported | supported | ≠ |"
    found = show_read(real_run, "W-97", "SRR1.2", tools=["igblast"])
    assert found["kind"] == "real" and list(found["tools"]) == ["igblast"]
    assert found["truth"]["known"] == ["IGHV3-23*01"] and found["tools"]["igblast"]["scores"]["v"] == "supported"
    assert found["read"]["sequence"] == "TTTTGGGGCC" and found["read"]["isotype"] == "IGHM"


def test_an_unknown_read_or_donor_is_refused(real_run, capsys):
    assert main(["show", "--run-dir", str(real_run.root), "--donor", "W-97", "--read", "SRR1.3"]) == 2
    err = capsys.readouterr().err
    assert "no read 'SRR1.3' of donor W-97" in err and "the nearest: SRR1.30, SRR1.2, SRR1.1" in err
    assert main(["show", "--run-dir", str(real_run.root), "--donor", "W-9", "--read", "SRR1.2"]) == 2
    err = capsys.readouterr().err
    assert "no donor 'W-9' in" in err and "donors: W-97" in err
    # a case and a read are one or the other
    for args in (["--case", "S0:a:1", "--donor", "W-97"], ["--donor", "W-97"], []):
        assert main(["show", "--run-dir", str(real_run.root), *args]) == 2
        assert "give --case ID, or --donor D and --read ID" in capsys.readouterr().err
