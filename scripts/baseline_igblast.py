"""IgBLAST baseline harness: the bar to beat.

Generates stratified GenAIRR records (clean -> CDR3 fragment), runs IgBLAST on the
SAME OGRDB germline the reads were simulated from, and scores its AIRR output against
ground truth with the same metrics we use for DNAlignAIR: per-gene top-1-in-set call
accuracy + in-sequence / germline start-end deviation, stratified by difficulty.
AIRR coords are 1-based; GenAIRR GT is 0-based (starts converted with -1).
"""
import argparse
import csv
import os
import subprocess
import tempfile

import numpy as np

import GenAIRR.data as gdata
from alignair.gym.gym import build_experiment
from alignair.gym.curriculum import Curriculum
from alignair.gym.crop import crop_record

TOOLS = os.path.join(os.path.dirname(__file__), "..", ".private", "tools")
IGB = os.path.join(TOOLS, "ncbi-igblast-1.22.0")
GENES = ("v", "d", "j")


def run_igblast(records: list) -> list:
    """Run igblastn (AIRR outfmt 19) on the records' sequences; return parsed rows
    aligned to input order by sequence_id."""
    with tempfile.TemporaryDirectory() as td:
        qpath = os.path.join(td, "q.fasta")
        with open(qpath, "w") as f:
            for i, r in enumerate(records):
                f.write(f">seq{i}\n{r['sequence']}\n")
        out = os.path.join(td, "out.tsv")
        env = dict(os.environ, IGDATA=IGB)
        cmd = [os.path.join(IGB, "bin", "igblastn"),
               "-germline_db_V", os.path.join(TOOLS, "germline", "igh_v"),
               "-germline_db_D", os.path.join(TOOLS, "germline", "igh_d"),
               "-germline_db_J", os.path.join(TOOLS, "germline", "igh_j"),
               "-auxiliary_data", os.path.join(IGB, "optional_file", "human_gl.aux"),
               "-organism", "human", "-ig_seqtype", "Ig", "-num_threads", "8",
               "-query", qpath, "-outfmt", "19", "-out", out]
        subprocess.run(cmd, env=env, check=True, capture_output=True)
        rows = {r["sequence_id"]: r for r in csv.DictReader(open(out), delimiter="\t")}
    return [rows.get(f"seq{i}") for i in range(len(records))]


def _f(v):
    try:
        return float(v) if v not in ("", None) else None
    except (ValueError, TypeError):
        return None


def igblast_to_pred(row) -> dict:
    """Convert an IgBLAST AIRR row to a GenAIRR-convention prediction dict
    (1-based AIRR start -> 0-based; end kept as position). Missing -> None."""
    p = {}
    for g in GENES:
        call = (row or {}).get(f"{g}_call", "") or ""
        p[f"{g}_call"] = call.split(",")[0] if call else None
        ss, se = _f((row or {}).get(f"{g}_sequence_start")), _f((row or {}).get(f"{g}_sequence_end"))
        gs, ge = _f((row or {}).get(f"{g}_germline_start")), _f((row or {}).get(f"{g}_germline_end"))
        p[f"{g}_sequence_start"] = (ss - 1) if ss is not None else None
        p[f"{g}_sequence_end"] = se
        p[f"{g}_germline_start"] = (gs - 1) if gs is not None else None
        p[f"{g}_germline_end"] = ge
    return p


def score(records: list, preds: list) -> dict:
    """Per-gene top-1-in-set call accuracy + in-seq/germline start-end MAE vs GT.
    ``preds`` are GenAIRR-convention dicts (use igblast_to_pred for IgBLAST rows)."""
    agg = {g: {"call": [], "gene": [], "srec": [], "sprec": [], "ssize": [],
               "ss": [], "se": [], "gs": [], "ge": []} for g in GENES}
    for rec, pred in zip(records, preds):
        for g in GENES:
            gt_call = rec.get(f"{g}_call")
            if not gt_call:
                continue
            gt_set = set(str(gt_call).split(","))
            gt_genes = {a.split("*")[0] for a in gt_set}     # gene = allele before '*'
            pred_top1 = (pred or {}).get(f"{g}_call")
            agg[g]["call"].append(1.0 if pred_top1 in gt_set else 0.0)
            pred_gene = pred_top1.split("*")[0] if pred_top1 else None
            agg[g]["gene"].append(1.0 if pred_gene in gt_genes else 0.0)
            # multi-label set metrics (if a predicted set is available)
            pset = set((pred or {}).get(f"{g}_call_set") or ([pred_top1] if pred_top1 else []))
            if pset:
                inter = len(pset & gt_set)
                agg[g]["srec"].append(inter / len(gt_set))
                agg[g]["sprec"].append(inter / len(pset))
                agg[g]["ssize"].append(len(pset))
            if rec.get(f"{g}_sequence_start") is None:
                continue
            for key, gt in (("ss", f"{g}_sequence_start"), ("se", f"{g}_sequence_end"),
                            ("gs", f"{g}_germline_start"), ("ge", f"{g}_germline_end")):
                v = (pred or {}).get(gt)
                if v is not None:
                    agg[g][key].append(abs(v - rec[gt]))
    out = {}
    for g in GENES:
        a = agg[g]
        out[g] = {"call": float(np.mean(a["call"])) if a["call"] else float("nan"),
                  "gene": float(np.mean(a["gene"])) if a["gene"] else float("nan"),
                  "found": len(a["ss"]) / max(len(a["call"]), 1),
                  "srec": float(np.mean(a["srec"])) if a["srec"] else float("nan"),
                  "sprec": float(np.mean(a["sprec"])) if a["sprec"] else float("nan"),
                  "ssize": float(np.mean(a["ssize"])) if a["ssize"] else float("nan")}
        for k in ("ss", "se", "gs", "ge"):
            out[g][k] = float(np.mean(a[k])) if a[k] else float("nan")
    return out


def gen_records(p: float, n: int, seed: int, crop_to: int | None,
                overrides: dict | None = None) -> list:
    params = Curriculum().params(p)
    if overrides:
        params.update(overrides)            # extreme strata (heavy SHM / trim beyond the ramp cap)
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, params)
    recs = list(exp.stream_records(n=n, seed=seed))
    if crop_to is not None:
        recs = [crop_record(r, crop_to) for r in recs]
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    strata = [("clean", 0.0, None), ("moderate", 0.5, None),
              ("hard", 1.0, None), ("fragment~80bp", 1.0, 80)]
    print(f"IgBLAST baseline | HUMAN_IGH_OGRDB | n={args.n} per stratum\n")
    for name, p, crop in strata:
        recs = gen_records(p, args.n, args.seed, crop)
        preds = [igblast_to_pred(r) for r in run_igblast(recs)]
        print(f"[{name}]")
        print_scores(score(recs, preds))
        print()


def print_scores(s: dict, indent: str = "  ", sets: bool = False) -> None:
    for g in GENES:
        r = s[g]
        line = (f"{indent}{g.upper()}: call={r['call']:.2f} gene={r['gene']:.2f} "
                f"found={r['found']:.2f} seq[{r['ss']:.1f},{r['se']:.1f}] gl[{r['gs']:.1f},{r['ge']:.1f}]")
        if sets and r.get("srec") == r.get("srec"):  # not NaN
            line += f" set[rec={r['srec']:.2f} prec={r['sprec']:.2f} sz={r['ssize']:.1f}]"
        print(line)


if __name__ == "__main__":
    main()
