"""Comparison reports cluster the reads of a clone and the two strands of a read (``scoring.clusters``), and a
comparison of two checkpoints counts the reads the second fixes and breaks."""

import dataclasses

import pytest
from alignair_assay.report import to_markdown
from alignair_assay.scoring.clusters import CLUSTER_KEYS
from alignair_assay.truth.model import ObservableTruth
from alignair_assay.workflow.scoring import ToolScores, comparison_report

pytestmark = pytest.mark.integration


def _truth(case_id: str, groups: dict) -> ObservableTruth:
    return ObservableTruth(case_id=case_id, stratum="lineages", sequence="ACGT", orientation=0, segments={},
                           unscorable=frozenset(), junction=None, vj_in_frame=None, stop_codon=None, productive=None,
                           clone_key=None, v_mismatches=None, v_frameshift=None, j_frameshift=None, excluded=None,
                           warnings=(), groups=groups)


def test_a_comparison_of_clonal_reads_is_clustered_by_clone(tmp_path):
    # 5 clones of 20 reads; b gets every read of the first clone right where a gets it wrong. A read-level test
    # would call that significant (p 2e-6); 5 clusters are too few for any verdict.
    truths = [_truth(f"S9:lineages:{k:06d}", {"clone": str(k // 20)}) for k in range(100)]
    a = ToolScores("a", {t.case_id: {"v.top1_in_set": 0.0 if k < 20 else 1.0} for k, t in enumerate(truths)}, {})
    b = ToolScores("b", {t.case_id: {"v.top1_in_set": 1.0} for t in truths}, {})
    truth_file = tmp_path / "truth.jsonl.gz"
    truth_file.write_bytes(b"")
    report = comparison_report(a, b, truths, {"reference_digest": "0" * 64}, command="compare",
                               inputs={"truth": truth_file}, reference_name="HUMAN_IGH_OGRDB")
    entry = report["comparison"]["v.top1_in_set"]
    assert (entry["n_paired"], entry["n_clusters"], entry["test"], entry["p_value"]) == (100, 5, "too_few_clusters",
                                                                                         1.0)
    assert entry["verdict"] == "tie_or_inconclusive"
    provenance = report["provenance"]
    assert provenance["cluster_keys"] == list(CLUSTER_KEYS)
    assert "clustered" in provenance["tests"] and "too_few_clusters" in provenance["tests"]
    md = to_markdown(report)
    assert "| v.top1_in_set | 80.0% (n=100, 100%) | 100.0% (n=100, 100%) | 100 (5 clusters) |" in md
    assert "| tie_or_inconclusive (too few clusters) |" in md and "fewer than 10 clusters" in md

    # Reads without a clone or pair are compared read by read, as before.
    plain = [_truth(t.case_id, {"subject": "donor1"}) for t in truths]
    report = comparison_report(a, b, plain, {"reference_digest": "0" * 64}, command="compare",
                               inputs={"truth": truth_file}, reference_name="HUMAN_IGH_OGRDB")
    entry = report["comparison"]["v.top1_in_set"]
    assert (entry["n_clusters"], entry["test"], entry["verdict"]) == (100, "mcnemar_exact", "b_better")
    md = to_markdown(report)
    assert "| 100 | " in md and "clusters" not in md


def test_a_comparison_of_two_checkpoints_counts_the_reads_fixed_and_broken(tmp_path):
    # b fixes the 20 reads of the first stratum a gets wrong and breaks 3 of the second's
    truths = [dataclasses.replace(_truth(f"S9:{'one' if k < 50 else 'two'}:{k:06d}", {}),
                                  stratum="one" if k < 50 else "two") for k in range(100)]
    a = ToolScores("x@base", {t.case_id: {"v.top1_in_set": 0.0 if k < 20 else 1.0} for k, t in enumerate(truths)}, {})
    b = ToolScores("x@cand", {t.case_id: {"v.top1_in_set": 0.0 if 50 <= k < 53 else 1.0}
                              for k, t in enumerate(truths)}, {})
    truth_file = tmp_path / "truth.jsonl.gz"
    truth_file.write_bytes(b"")
    args = dict(command="compare", inputs={"truth": truth_file}, reference_name="HUMAN_IGH_OGRDB")
    assert "changes" not in comparison_report(a, b, truths, {"reference_digest": "0" * 64}, **args)
    report = comparison_report(a, b, truths, {"reference_digest": "0" * 64}, **args, changes=True)
    reads = report["changes"]["reads"]
    assert (reads["fixed"], reads["broken"]) == (20, 3)
    assert reads["fixed_examples"] == [f"S9:one:{k:06d}" for k in range(0, 20, 2)]  # evenly spaced
    assert reads["groups"] == {"one": {"fixed": 20, "broken": 0}, "two": {"fixed": 0, "broken": 3}}
    md = to_markdown(report)
    section = md[md.index("\n## Reads fixed and broken\n"):]
    assert "x@cand fixes 20 reads x@base gets wrong and breaks 3 it gets right" in section
    assert "| one | 20 | 0 |" in section and "| two | 0 | 3 |" in section
    assert "| v.top1_in_set | 100 | 20 | 3 | S9:one:000000, S9:one:000002" in section
