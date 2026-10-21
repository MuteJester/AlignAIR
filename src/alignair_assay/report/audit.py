"""The readiness audit's markdown (``audit.md``), rendered from the plain record ``workflow.audit.run_audit`` writes.

The first line is the verdict (``AUDIT PASS: ...`` or ``AUDIT FAIL: ...``), then what was audited, one table row per
check (status, n and its evidence's summary; a failure in bold), and per group of checks each one's reason and its
evidence in full.
"""
from __future__ import annotations

import json

from alignair_assay.report.markdown import MISSING, markdown_table

_STATUS = {"fail": "**fail**"}


def audit_markdown(record: dict) -> str:
    """``audit.md`` of an audit record (``workflow.audit.run_audit``): the verdict's line first."""
    config, checkpoint, provenance = record.get("config") or {}, record.get("checkpoint") or {}, \
        record.get("provenance") or {}
    assay, genairr = provenance.get("assay") or {}, provenance.get("genairr") or {}
    frozen = provenance.get("genairr_freeze")
    if frozen:
        frozen_text = (f", a frozen copy of commit `{(frozen.get('genairr_git_commit') or MISSING)[:12]}`"
                       f"{' (dirty)' if frozen.get('genairr_git_dirty') else ''} from {frozen.get('source')}")
    else:
        frozen_text = ", **live** (not a frozen copy: a GenAIRR commit while the audit ran would stale its builds)"
    commit = assay.get("git_commit")
    lines = [record["verdict"]["line"], "", "# AlignAIR assay audit", "",
             "The readiness gate for model comparisons: the truth checks itself, the outputs reproduce, one "
             "checkpoint compared with itself differs by nothing on every surface, and IgBLAST lands where "
             "independent evidence says it should. A warning asks for a look and never fails the audit; a skipped "
             "check says why it could not be checked now.", "",
             f"- Checkpoint `{checkpoint.get('path')}` (`{(checkpoint.get('sha256') or '')[:12]}`); the null "
             f"comparisons ran on {config.get('device')}.",
             f"- Stored runs, read only: `{config.get('sim_run')}` (all) and `{config.get('real_run')}` (real); data "
             f"root `{config.get('data_root')}`. Anything re-run is under `audit/scratch/`.",
             f"- Assay commit `{(commit or MISSING)[:12]}`{' (dirty)' if assay.get('git_dirty') else ''}, code "
             f"`{(provenance.get('audit_code_sha256') or MISSING)[:12]}`; GenAIRR "
             f"{genairr.get('genairr_version')} (`{(genairr.get('genairr_source_sha256') or '')[:12]}`"
             f"{frozen_text}); host {provenance.get('host')}; {record.get('created_utc')}, {record.get('seconds')} s.",
             ""]
    rows = [[found["name"], _STATUS.get(found["status"], found["status"]), f"{found['n']:,}",
             _cell(found["evidence"].get("summary"))] for found in record["checks"]]
    lines += markdown_table(["Check", "Status", "n", "Evidence"], rows)
    groups = record.get("groups") or {}
    for group, title in groups.items():
        members = [found for found in record["checks"] if found["name"].split(".")[0] == group]
        if not members:
            continue
        lines += ["", f"## {title}"]
        for found in members:
            lines += ["", f"### {found['name']}: {found['status']} (n = {found['n']:,})", "",
                      found["evidence"].get("summary") or MISSING]
            if found.get("reason"):
                lines += ["", f"Reason: {found['reason']}"]
            evidence = {key: value for key, value in found["evidence"].items() if key != "summary"}
            if evidence:
                lines += ["", "```json", json.dumps(evidence, indent=1, sort_keys=True), "```"]
    return "\n".join(lines) + "\n"


def _cell(text) -> str:
    """A table cell: the text (``markdown_table`` escapes its pipes), or a dash."""
    return MISSING if not text else str(text)
