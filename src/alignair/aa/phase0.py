"""Simulation-driven Phase-0 report builder: stream GenAIRR IGH records, project codon
provenance, and compute the identifiability / length / boundary / D audits."""
from __future__ import annotations

import statistics

from .translation import frame_offset_of, project_codons
from .reference import build_aa_germlines
from .equivalence import db_hash, build_equivalence_classes
from .audit import identifiability_summary, prefix_identifiability_curve
from .report import assemble_report

_SEG_COLS = {
    "V": ("v_sequence_start", "v_sequence_end"),
    "D": ("d_sequence_start", "d_sequence_end"),
    "J": ("j_sequence_start", "j_sequence_end"),
}


def _segments(rec) -> dict:
    segs = {}
    for seg, (a, b) in _SEG_COLS.items():
        s, e = rec.get(a), rec.get(b)
        if s is not None and e is not None:
            segs[seg] = (int(s), int(e))
    return segs


def build_phase0_report(n: int, seed: int = 0, rate: float = 0.05,
                        productive_only: bool = True,
                        dc_name: str = "HUMAN_IGH_OGRDB") -> dict:
    import GenAIRR as ga

    dc = getattr(ga, dc_name)

    # ---- germline-derived identifiability (no simulation needed) ----
    gset = build_aa_germlines(dc, genes=("V", "J"))
    dbh = db_hash(
        [(a.name, a.ungapped_seq) for a in dc.allele_list("v")]
        + [(a.name, a.ungapped_seq) for a in dc.allele_list("d")]
        + [(a.name, a.ungapped_seq) for a in dc.allele_list("j")]
    )
    identifiability = {}
    coverage_curve = {}
    for gene in ("V", "J"):
        aa_by_name = {n_: g.aa_seq for n_, g in gset.genes[gene].items()}
        cmap = build_equivalence_classes(gene, aa_by_name, dbh)
        identifiability[gene] = identifiability_summary(cmap)
        if gene == "V":
            lengths = [30, 60, 90, 120, max((len(s) for s in aa_by_name.values()), default=0)]
            coverage_curve[gene] = prefix_identifiability_curve(aa_by_name, lengths)

    # ---- simulation-derived stats ----
    exp = ga.Experiment.on(dc_name).recombine().mutate(model="s5f", rate=rate)
    aa_lengths = []
    n_x = n_stop = n_aa_chars = 0
    v_end_cuts = j_start_cuts = 0
    owned_d = []
    zero_d = 0
    used = 0
    for rec in exp.stream_records(n=n, seed=seed):
        js = rec.get("junction_start")
        aa = rec.get("sequence_aa")
        if js is None or not aa:
            continue
        if productive_only and not rec.get("productive"):
            continue
        used += 1
        aa_lengths.append(len(aa))
        n_x += aa.count("X")
        n_stop += aa.count("*")
        n_aa_chars += len(aa)
        fo = frame_offset_of(js)
        segs = _segments(rec)
        proj = project_codons(len(rec["sequence"]), fo, segs)
        if "V" in proj.phase and proj.phase["V"]["end"] != 0:
            v_end_cuts += 1
        if "J" in proj.phase and proj.phase["J"]["start"] != 0:
            j_start_cuts += 1
        db = proj.aa_bounds.get("D")
        owned = (db[1] - db[0]) if db else 0
        owned_d.append(owned)
        if owned == 0:
            zero_d += 1

    def _pct(vals, q):
        if not vals:
            return 0
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))]

    length_stats = {
        "n_records": used,
        "aa_len_p50": _pct(aa_lengths, 0.50),
        "aa_len_p95": _pct(aa_lengths, 0.95),
        "aa_len_max": max(aa_lengths) if aa_lengths else 0,
    }
    residue_stats = {
        "frac_X": (n_x / n_aa_chars) if n_aa_chars else 0.0,
        "frac_stop": (n_stop / n_aa_chars) if n_aa_chars else 0.0,
    }
    boundary_stats = {
        "V_end_cut_frac": (v_end_cuts / used) if used else 0.0,
        "J_start_cut_frac": (j_start_cuts / used) if used else 0.0,
    }
    d_stats = {
        "mean_owned_d_residues": (statistics.fmean(owned_d) if owned_d else 0.0),
        "frac_zero_owned_d": (zero_d / used) if used else 0.0,
    }

    return assemble_report(
        meta={"dc": dc_name, "n": n, "seed": seed, "rate": rate,
              "productive_only": productive_only, "db_hash": dbh,
              "quarantined_v": len(gset.quarantined["V"]),
              "quarantined_j": len(gset.quarantined["J"])},
        identifiability=identifiability,
        length_stats=length_stats,
        residue_stats=residue_stats,
        boundary_stats=boundary_stats,
        d_stats=d_stats,
        coverage_curve=coverage_curve,
    )
