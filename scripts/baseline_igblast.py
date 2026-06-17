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


def _f(row, key):
    v = row.get(key, "") if row else ""
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def score(records: list, rows: list) -> dict:
    """Per-gene call accuracy + in-seq/germline start-end MAE vs ground truth."""
    agg = {g: {"call": [], "ss": [], "se": [], "gs": [], "ge": []} for g in GENES}
    for rec, row in zip(records, rows):
        for g in GENES:
            gt_call = rec.get(f"{g}_call")
            if not gt_call:
                continue
            gt_set = set(str(gt_call).split(","))
            pred_call = (row or {}).get(f"{g}_call", "") or ""
            pred_top1 = pred_call.split(",")[0] if pred_call else ""
            agg[g]["call"].append(1.0 if pred_top1 in gt_set else 0.0)
            # coordinates (AIRR 1-based start -> 0-based)
            ps, pe = _f(row, f"{g}_sequence_start"), _f(row, f"{g}_sequence_end")
            gs, ge = _f(row, f"{g}_germline_start"), _f(row, f"{g}_germline_end")
            if rec.get(f"{g}_sequence_start") is not None:
                if ps is not None:
                    agg[g]["ss"].append(abs((ps - 1) - rec[f"{g}_sequence_start"]))
                if pe is not None:
                    agg[g]["se"].append(abs(pe - rec[f"{g}_sequence_end"]))
                if gs is not None:
                    agg[g]["gs"].append(abs((gs - 1) - rec[f"{g}_germline_start"]))
                if ge is not None:
                    agg[g]["ge"].append(abs(ge - rec[f"{g}_germline_end"]))
    out = {}
    for g in GENES:
        a = agg[g]
        out[g] = {
            "call": float(np.mean(a["call"])) if a["call"] else float("nan"),
            "found": len(a["ss"]) / max(len(a["call"]), 1),  # fraction IgBLAST located
            "ss": float(np.mean(a["ss"])) if a["ss"] else float("nan"),
            "se": float(np.mean(a["se"])) if a["se"] else float("nan"),
            "gs": float(np.mean(a["gs"])) if a["gs"] else float("nan"),
            "ge": float(np.mean(a["ge"])) if a["ge"] else float("nan"),
        }
    return out


def gen_records(p: float, n: int, seed: int, crop_to: int | None) -> list:
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, Curriculum().params(p))
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
        rows = run_igblast(recs)
        s = score(recs, rows)
        print(f"[{name}]")
        for g in GENES:
            r = s[g]
            print(f"  {g.upper()}: call={r['call']:.2f} found={r['found']:.2f} "
                  f"seq[{r['ss']:.1f},{r['se']:.1f}] gl[{r['gs']:.1f},{r['ge']:.1f}]")
        print()


if __name__ == "__main__":
    main()
