"""Assemble + render the frozen Phase-0 diagnostic report (no simulation here)."""
from __future__ import annotations

import json
import os


def assemble_report(*, meta, identifiability, length_stats, residue_stats,
                    boundary_stats, d_stats, coverage_curve) -> dict:
    return {
        "schema": "alignair.aa.phase0.v1",
        "meta": meta,
        "identifiability": identifiability,
        "length_stats": length_stats,
        "residue_stats": residue_stats,
        "boundary_stats": boundary_stats,
        "d_stats": d_stats,
        "coverage_curve": coverage_curve,
    }


def render_markdown(report: dict) -> str:
    m = report["meta"]
    idn = report["identifiability"]
    ls = report["length_stats"]
    rs = report["residue_stats"]
    bs = report["boundary_stats"]
    ds = report["d_stats"]
    lines = [
        "# Amino-Acid AlignAIR — Phase 0 Diagnostic Report",
        "",
        f"- Dataconfig: `{m.get('dc')}`  |  simulated records: {m.get('n')}  |  seed: {m.get('seed')}",
        f"- Germline DB hash: `{m.get('db_hash')}`",
        "",
        "## Identifiability (go/no-go gate)",
    ]
    for gene in ("V", "J"):
        g = idn.get(gene, {})
        lines.append(
            f"- {gene}: {g.get('n_classes')}/{g.get('n_alleles')} protein classes; "
            f"uniquely identifiable fraction = {g.get('frac_alleles_uniquely_identifiable')}"
        )
    lines += [
        "",
        "## Sequence & residue audit",
        f"- AA length p50/p95: {ls.get('aa_len_p50')}/{ls.get('aa_len_p95')} "
        f"over {ls.get('n_records')} records",
        f"- unknown-residue fraction: {rs.get('frac_X')}  |  stop fraction: {rs.get('frac_stop')}",
        "",
        "## Boundary & D-segment audit",
        f"- V-end boundary cuts a codon in {bs.get('V_end_cut_frac')} of reads; "
        f"J-start in {bs.get('J_start_cut_frac')}",
        f"- mean fully-owned D residues: {ds.get('mean_owned_d_residues')}; "
        f"reads with 0 owned D residues: {ds.get('frac_zero_owned_d')}",
    ]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_markdown(report))
    return out_dir
